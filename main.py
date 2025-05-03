# Import functions from the Ising_Model_Fast module
from Ising_Model_Fast import *
import time
import numpy as np
import matplotlib.pyplot as plt

# ===========================================
#       PARAMETERS AND CONFIGURATION
# ===========================================
seed = 3500
np.random.seed(seed)
test_verbose = True

# Simulation parameters for single-T run (images + time evolution)
N = 100
J1, J2 = 1.0, 0.0
T_fixed = 3.5
MC_steps = 10_000
Iterations = N*N*MC_steps

# Time testing markers
t_inits_i = time.time()

# ===========================================
#       INITIALIZATIONS AND SETUP
# ===========================================
# Lattice and energy for single-T run
tilattice = initialize_lattice(N, p=0.5, seed=seed)
initial_energy = get_energy(tilattice, N, J1, J2)
# Paths for saving
paths = path_configuration(N, T_fixed)

t_inits_f = time.time()
t_simul_i = time.time()

# ===========================================
#         SINGLE-T SIMULATION
# ===========================================
simulation_params = {
    'MC_steps': Iterations,
    'T': T_fixed,
    'N': N,
    'J1': J1,
    'J2': J2,
    'save_images': True,
    'images_spacing': np.unique(np.logspace(0, np.log10(MC_steps), 100, dtype=int)),
    'seed': seed,
}
# Run Metropolis once for images, spins and energies
spins, energies, images, last_config = metropolis(
    lattice=tilattice,
    energy=initial_energy,
    **simulation_params
)
use_last = 10000 if Iterations > 10000 else int(0.5 * Iterations)

mean_M = np.mean(spins[-use_last:])
std_M = np.std(spins[-use_last:], ddof=1)
mean_E = np.mean(energies[-use_last:])
std_E = np.std(energies[-use_last:], ddof=1)
C_v = np.var(energies[-use_last:], ddof=1) / (T_fixed**2 * N * N)
std_C = std_E / (T_fixed**2 * N * N)

with open(
    paths['data'] / f'single_T_statistics_N{N}_T{T_fixed}.txt',
    'w', encoding='utf-8'
) as f:
    f.write(f"# Single temperature analysis for T = {T_fixed}, N = {N}\n")
    f.write("Quantity\tMean\tStdError\n")
    f.write(f"Magnetization\t{mean_M:.6f}\t{std_M:.6f}\n")
    f.write(f"Energy\t{mean_E:.6f}\t{std_E:.6f}\n")
    f.write(f"Specific Heat\t{C_v:.6f}\t{std_C:.6f}\n")
t_simul_f = time.time()

# Save time-series data and images
fast_save_data(spins, paths['data'], 'spins')
fast_save_data(energies, paths['data'], 'energies')
fast_save_data(last_config, paths['data'], f'last_config_N{N}_T{T_fixed}_MC{MC_steps}')
create_gif(images, save_dir=paths['figures'], filename='demo.gif', scale=5, fps=15, cmap='plasma')
save_images_as_png(images, save_dir=paths['images'], prefix='ising', cmap='plasma', scale=5)

# Plot time-series
def plot_time_series(data, ylabel, title, fname):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Paso Monte Carlo')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(paths['figures'] / fname)
    plt.close()

plot_time_series(spins, 'Magnetization', 'Magnetization vs MC steps', 'magnetizacion_vs_pasos.png')
plot_time_series(energies, 'Energy', 'Energy vs MC steps', 'energia_vs_pasos.png')




# ===========================================
#     BARRIDO EN TEMPERATURAS PARA VARIOS N
# ===========================================
Ts = get_clustered_temperatures(n_temperatures=100, center=2.26, low=0.5, high=4)
#Ts = np.linspace(2,7,100)
N_values = np.array([64, 128, 256, 512], dtype=int)
MC_steps_temp = 50_000

Tc_estimates = np.empty_like(N_values, dtype=np.float32)
beta_estimates = np.empty_like(N_values, dtype=np.float32)
alpha_estimates = np.empty_like(N_values, dtype=np.float32)
Tc_errors = np.empty_like(N_values, dtype=np.float32)
beta_errors = np.empty_like(N_values, dtype=np.float32)
alpha_errors = np.empty_like(N_values, dtype=np.float32)
gamma_estimates = np.empty_like(N_values, dtype=np.float32)
gamma_errors = np.empty_like(N_values, dtype=np.float32)




for i, Ni in enumerate(N_values):
    Ni_int = int(Ni)
    # Inicializar red y energía para este N
    lattice = initialize_lattice(Ni_int, p=0.75, seed=seed)
    E0 = get_energy_fast(lattice, Ni_int, J1, J2)  # Usar función optimizada
    
    # Barrido de temperaturas usando la función Numba
    avg_mags, std_mags, avg_energies, std_energies, heat_capacities, std_Cv, susceptibilities, std_chi = get_M_E_C_of_T_numba(
        lattice=lattice,
        energy=E0,
        Ts=Ts,
        N=Ni_int,
        J1=J1,
        J2=J2,
        MC_steps=MC_steps_temp,
        seed=seed,
        use_last=10000
    )
    

    # Save thermal averages
    out_txt = paths['data'] / f'promedios_T_N{Ni_int}.txt'
    with open(out_txt, 'w') as f:
        f.write('T\tM_avg\tM_err\tE_avg\tE_err\tCv\tCv_err\tChi\tChi_err\n')
        for T_val, m, dm, e, de, c, dc, chi, dchi in zip(Ts, avg_mags, std_mags, avg_energies, std_energies, heat_capacities, std_Cv, susceptibilities, std_chi):
            f.write(f"{T_val:.3f}\t{m:.6f}\t{dm:.6f}\t{e:.6f}\t{de:.6f}\t{c:.6f}\t{dc:.6f}\t{chi:.6f}\t{dchi:.6f}\n")


    # Generate and save plots
    plot_quantity_vs_T(Ts, avg_mags, errors=std_mags,
                   ylabel='Average Magnetization',
                   title=f'Magnetization vs Temperature (N={Ni_int})',
                   save_path=paths['figures'] / f'magnetization_vs_T_N{Ni_int}.png',
                   color='red', connect_points=False)

    plot_quantity_vs_T(Ts, avg_energies, errors=std_energies,
                       ylabel='Average Energy',
                       title=f'Energy vs Temperature (N={Ni_int})',
                       save_path=paths['figures'] / f'energy_vs_T_N{Ni_int}.png',
                       color='green', connect_points=False)

    plot_quantity_vs_T(Ts, heat_capacities, errors=std_Cv,
                       ylabel='Specific Heat $c_v$',
                       title=f'Specific Heat vs Temperature (N={Ni_int})',
                       save_path=paths['figures'] / f'Cv_vs_T_N{Ni_int}.png',
                       color='blue', connect_points=False)
    plot_quantity_vs_T(Ts, susceptibilities, errors=std_chi,
                   ylabel='Susceptibility $\\chi$',
                   title=f'Susceptibility vs Temperature (N={Ni_int})',
                   save_path=paths['figures'] / f'susceptibility_vs_T_N{Ni_int}.png',
                   color='darkviolet', connect_points=False)
                    
    
    # Estimate Tc for this N
    Tc_i, Tc_err = find_Tc(Ts, heat_capacities, std_Cv)
    Tc_estimates[i] = Tc_i
    Tc_errors[i] = Tc_err

    print(f"N={Ni_int} → Tc ≈ {Tc_i:.3f} ± {Tc_err:.3f}")
    with open(
    paths['data'] / f'Tc_N{Ni_int}.txt',
    'w', encoding='utf-8'
    ) as f:
        f.write(f"Tc = {Tc_i:.6f} ± {Tc_err:.6f}\n")
        
        
    # === Estimar exponentes críticos β y α cerca de Tc ===
    # === Estimar exponentes críticos β, α y γ cerca de Tc ===
    beta_fit, beta_err, alpha_fit, alpha_err, gamma_fit, gamma_err, mask_critical = estimate_critical_exponents(
        Ts,
        avg_mags, std_mags,
        heat_capacities, std_Cv,
        susceptibilities, std_chi,
        Tc_estimates[i]
    )
    
    if not np.isnan(beta_fit):
        print(f"[N={Ni_int}] Critical exponents near Tc ≈ {Tc_estimates[i]:.3f}")
        print(f"\tβ ≈ {beta_fit:.3f} ± {beta_err:.3f}")
        print(f"\tα ≈ {alpha_fit:.3f} ± {alpha_err:.3f}")
        print(f"\tγ ≈ {gamma_fit:.3f} ± {gamma_err:.3f}")
    
        # Guardar valores
        beta_estimates[i] = beta_fit
        alpha_estimates[i] = alpha_fit
        gamma_estimates[i] = gamma_fit
        beta_errors[i] = beta_err
        alpha_errors[i] = alpha_err
        gamma_errors[i] = gamma_err
    
        # Guardar a archivo
        with open(paths['data'] / f'exponentes_N{Ni_int}.txt', 'w', encoding='utf-8') as f:
            f.write(f"beta = {beta_fit:.6f} ± {beta_err:.6f}\n")
            f.write(f"alpha = {alpha_fit:.6f} ± {alpha_err:.6f}\n")
            f.write(f"gamma = {gamma_fit:.6f} ± {gamma_err:.6f}\n")
    
        # === Preparar datos log-log ===
        log_T = np.log(Tc_estimates[i] - Ts[mask_critical])
        log_M = np.log(avg_mags[mask_critical] + 1e-10)
        log_C = np.log(heat_capacities[mask_critical] + 1e-10)
        log_chi = np.log(susceptibilities[mask_critical] + 1e-10)
    
        log_M_errors = std_mags[mask_critical] / (avg_mags[mask_critical] + 1e-10)
        log_C_errors = std_Cv[mask_critical] / (heat_capacities[mask_critical] + 1e-10)
        log_chi_errors = std_chi[mask_critical] / (susceptibilities[mask_critical] + 1e-10)
    
        weights_M = 1.0 / (log_M_errors**2 + 1e-12)
        weights_C = 1.0 / (log_C_errors**2 + 1e-12)
        weights_chi = 1.0 / (log_chi_errors**2 + 1e-12)
    
        # Ajustes
        beta_fit_line, _ = np.polyfit(log_T, log_M, 1, w=weights_M, cov=True)
        alpha_fit_line, _ = np.polyfit(log_T, log_C, 1, w=weights_C, cov=True)
        gamma_fit_line, _ = np.polyfit(log_T, log_chi, 1, w=weights_chi, cov=True)
    
        fit_line_M = np.polyval(beta_fit_line, log_T)
        fit_line_C = np.polyval(alpha_fit_line, log_T)
        fit_line_chi = np.polyval(gamma_fit_line, log_T)
    
        # === Plot β ===
        plt.figure(figsize=(6, 4))
        plt.errorbar(log_T, log_M, yerr=log_M_errors, fmt='o', markersize=4,
                     color='darkorange', label='Simulated data', capsize=3)
        plt.plot(log_T, fit_line_M, 'r--', label=f'Fit (β ≈ {beta_fit_line[0]:.3f})')
        plt.xlabel(r'$\log(T_c - T)$')
        plt.ylabel(r'$\log(M)$')
        plt.title(r'Log-log fit for $\beta$ (N = %d)' % Ni_int)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths['figures'] / f'loglog_beta_N{Ni_int}.png', dpi=300)
        plt.close()
    
        # === Plot α ===
        plt.figure(figsize=(6, 4))
        plt.errorbar(log_T, log_C, yerr=log_C_errors, fmt='o', markersize=4,
                     color='royalblue', label='Simulated data', capsize=3)
        plt.plot(log_T, fit_line_C, 'r--', label=f'Fit (α ≈ {-alpha_fit_line[0]:.3f})')
        plt.xlabel(r'$\log(T_c - T)$')
        plt.ylabel(r'$\log(c_v)$')
        plt.title(r'Log-log fit for $\alpha$ (N = %d)' % Ni_int)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths['figures'] / f'loglog_alpha_N{Ni_int}.png', dpi=300)
        plt.close()
    
        # === Plot γ ===
        plt.figure(figsize=(6, 4))
        plt.errorbar(log_T, log_chi, yerr=log_chi_errors, fmt='o', markersize=4,
                     color='darkviolet', label='Simulated data', capsize=3)
        plt.plot(log_T, fit_line_chi, 'r--', label=f'Fit (γ ≈ {-gamma_fit_line[0]:.3f})')
        plt.xlabel(r'$\log(T_c - T)$')
        plt.ylabel(r'$\log(\chi)$')
        plt.title(r'Log-log fit for $\gamma$ (N = %d)' % Ni_int)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths['figures'] / f'loglog_gamma_N{Ni_int}.png', dpi=300)
        plt.close()

        
        
# ===========================================
#   EXTRAPOLACIÓN AL LÍMITE TERMODINÁMICO
# ===========================================
Tc_inf, Tc_err, Tc_coef = extrapolate_Tc(N_values, Tc_estimates)
print(f"Estimación Tc(∞) = {Tc_inf:.3f} ± {Tc_err:.3f}")
# Plot extrapolation
invN = 1.0 / N_values
plt.figure(figsize=(6, 4))
plt.errorbar(invN, Tc_estimates, yerr=Tc_errors, fmt='o', color='black', label='Tc(N)', capsize=3)
plt.plot(invN, Tc_coef[0]*invN + Tc_coef[1],
         'r--', label=f'Fit: Tc = {Tc_coef[0]:.2f}/N + {Tc_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel('Critical temperature $T_c$')
plt.title(f'Extrapolation of $T_c$ to thermodynamic limit')
plt.grid(False)
plt.tight_layout()
plt.legend()
plt.ylim(2, 6)
plt.savefig(paths['figures'] / 'Tc_extrapolation.png', dpi=300)
plt.close()

# ===========================================
#   EXTRAPOLACIÓN DE β, α Y γ AL LÍMITE N → ∞
# ===========================================
invN = 1.0 / N_values

# Extrapolar β
beta_inf, beta_err, beta_coef = extrapolate_exponent(N_values, beta_estimates, beta_errors, label='β')
plt.figure(figsize=(6, 4))
plt.errorbar(invN, beta_estimates, yerr=beta_errors, fmt='o', color='purple', label=r'$\beta(N)$', capsize=3)
plt.plot(invN, beta_coef[0]*invN + beta_coef[1],
         'r--', label=f'Fit: β = {beta_coef[0]:.2f}/N + {beta_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel(r'Critical exponent $\beta$')
plt.title(r'Extrapolation of $\beta$ to thermodynamic limit')
plt.grid(False)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'beta_extrapolation.png', dpi=300)
plt.close()

# Extrapolar α
alpha_inf, alpha_err, alpha_coef = extrapolate_exponent(N_values, alpha_estimates, alpha_errors, label='α')
plt.figure(figsize=(6, 4))
plt.errorbar(invN, alpha_estimates, yerr=alpha_errors, fmt='o', color='blue', label=r'$\alpha(N)$', capsize=3)
plt.plot(invN, alpha_coef[0]*invN + alpha_coef[1],
         'r--', label=f'Fit: α = {alpha_coef[0]:.2f}/N + {alpha_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel(r'Critical exponent $\alpha$')
plt.title(r'Extrapolation of $\alpha$ to thermodynamic limit')
plt.grid(False)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'alpha_extrapolation.png', dpi=300)
plt.close()

# Extrapolar γ
gamma_inf, gamma_err, gamma_coef = extrapolate_exponent(N_values, gamma_estimates, gamma_errors, label='γ')
plt.figure(figsize=(6, 4))
plt.errorbar(invN, gamma_estimates, yerr=gamma_errors, fmt='o', color='darkviolet',
             label=r'$\gamma(N)$', capsize=3)
plt.plot(invN, gamma_coef[0]*invN + gamma_coef[1],
         'r--', label=f'Fit: γ = {gamma_coef[0]:.2f}/N + {gamma_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel(r'Critical exponent $\gamma$')
plt.title(r'Extrapolation of $\gamma$ to thermodynamic limit')
plt.grid(False)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'gamma_extrapolation.png', dpi=300)
plt.close()

# Guardar resultados extrapolados
with open(
    paths['data'] / 'extrapolaciones_infinito.txt',
    'w', encoding='utf-8'
) as f:
    f.write(f"Tc(∞) = {Tc_inf:.6f} ± {Tc_err:.6f}\n")
    f.write(f"β(∞) = {beta_inf:.6f} ± {beta_err:.6f}\n")
    f.write(f"α(∞) = {alpha_inf:.6f} ± {alpha_err:.6f}\n")
    f.write(f"γ(∞) = {gamma_inf:.6f} ± {gamma_err:.6f}\n")

# Final time testing
t_save_data_f = time.time()
t_save_gif_f = time.time()
if test_verbose:
    print('\n===========================================')
    print('Time testing results:')
    print(f"\tInitialization time: {t_inits_f - t_inits_i:.2f} s")
    print(f"\tSingle-T simulation time: {t_simul_f - t_simul_i:.2f} s")
    print(f"\tData & figure saving time: {t_save_data_f - t_simul_f:.2f} s")
    print(f"\tGIF saving time: {t_save_gif_f - t_save_data_f:.2f} s")
    print('===========================================\n')
