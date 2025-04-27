# Import functions from the Ising_Model_Fast module
from Ising_Model_Fast import *
import time
import numpy as np
import matplotlib.pyplot as plt

# ===========================================
#       PARAMETERS AND CONFIGURATION
# ===========================================
seed = 3025
np.random.seed(seed)
test_verbose = True
paths = path_configuration(N=2, T=2)


# ===========================================
#     BARRIDO EN TEMPERATURAS PARA VARIOS N
# ===========================================
N_values = np.array([64, 128, 256, 512], dtype=int)
Ts = get_clustered_temperatures(n_temperatures=50, center=2.26, low=1.0, high=4)
J1, J2 = 1.0, 0.0
MC_steps_temp = 50_000

Tc_estimates = np.empty_like(N_values, dtype=np.float32)
beta_estimates = np.empty_like(N_values, dtype=np.float32)
alpha_estimates = np.empty_like(N_values, dtype=np.float32)
Tc_errors = np.empty_like(N_values, dtype=np.float32)
beta_errors = np.empty_like(N_values, dtype=np.float32)
alpha_errors = np.empty_like(N_values, dtype=np.float32)



for i, Ni in enumerate(N_values):
    Ni_int = int(Ni)
    print(f"\n\n====================\nN = {Ni_int}\n====================")
    # Inicializar red y energía para este N
    lattice = initialize_lattice(Ni_int, p=0.75, seed=seed)
    E0 = get_energy_fast(lattice, Ni_int, J1, J2)  # Usar función optimizada
    
    # Barrido de temperaturas usando la función Numba
    avg_mags, std_mags, avg_energies, std_energies, heat_capacities, std_Cv = get_M_E_C_of_T_numba(
        lattice=lattice,
        energy=E0,
        Ts=Ts,
        N=Ni_int,
        J1=J1,
        J2=J2,
        MC_steps=MC_steps_temp,
        seed=seed,
        use_last=5_000
    )
    
    avg_energies = avg_energies / (Ni_int**2)
    std_energies = std_energies / (Ni_int**2)

    # Save thermal averages
    out_txt = paths['data'] / f'promedios_T_N{Ni_int}.txt'
    with open(out_txt, 'w') as f:
        f.write('T\tM_avg\tM_err\tE_avg\tE_err\tCv\tCv_err\n')
        for T_val, m, dm, e, de, c, dc in zip(Ts, avg_mags, std_mags, avg_energies, std_energies, heat_capacities, std_Cv):
            f.write(f"{T_val:.3f}\t{m:.6f}\t{dm:.6f}\t{e:.6f}\t{de:.6f}\t{c:.6f}\t{dc:.6f}\n")


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
                       ylabel='Specific Heat $C_v$',
                       title=f'Specific Heat vs Temperature (N={Ni_int})',
                       save_path=paths['figures'] / f'Cv_vs_T_N{Ni_int}.png',
                       color='blue', connect_points=False)
    
    
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
    # === Estimar exponentes críticos β y α cerca de Tc ===
        beta_fit, beta_err, alpha_fit, alpha_err, mask_critical = estimate_critical_exponents(
            Ts, avg_mags, std_mags,
            heat_capacities, std_Cv,
            Tc_estimates[i]
            )

    # Solo continuar si la estimación fue válida
    if not np.isnan(beta_fit):
        print(f"[N={Ni_int}] Critical exponents near Tc ≈ {Tc_estimates[i]:.3f}")
        print(f"\tβ ≈ {beta_fit:.3f} ± {beta_err:.3f}")
        print(f"\tα ≈ {alpha_fit:.3f} ± {alpha_err:.3f}")
    
        # Guardar valores estimados y sus errores
        beta_estimates[i] = beta_fit
        alpha_estimates[i] = alpha_fit
        beta_errors[i] = beta_err
        alpha_errors[i] = alpha_err
    
        with open(
            paths['data'] / f'exponentes_N{Ni_int}.txt',
            'w', encoding='utf-8'
        ) as f:
            f.write(f"beta = {beta_fit:.6f} ± {beta_err:.6f}\n")
            f.write(f"alpha = {alpha_fit:.6f} ± {alpha_err:.6f}\n")
    
        # === Preparar log-log para graficar ===
        # === Preparar log-log para graficar ===
        Tc_i = Tc_estimates[i]
        log_T = np.log(Tc_i - Ts[mask_critical])
        log_M = np.log(avg_mags[mask_critical] + 1e-10)
        log_C = np.log(heat_capacities[mask_critical] + 1e-10)
        
        # Errores propagados: Δlog(x) ≈ Δx / x
        M_errors = std_mags[mask_critical]
        log_M_errors = M_errors / (avg_mags[mask_critical] + 1e-10)
        
        C_errors = std_Cv[mask_critical]
        log_C_errors = C_errors / (heat_capacities[mask_critical] + 1e-10)
        
        # === Ajustes ponderados ===
        weights_M = 1.0 / (log_M_errors**2 + 1e-12)
        weights_C = 1.0 / (log_C_errors**2 + 1e-12)
        
        beta_fit_line, _ = np.polyfit(log_T, log_M, 1, w=weights_M, cov=True)
        fit_line = np.polyval(beta_fit_line, log_T)
        
        alpha_fit_line, _ = np.polyfit(log_T, log_C, 1, w=weights_C, cov=True)
        fit_line_C = np.polyval(alpha_fit_line, log_T)
        
        # === Graficar log(M) vs log(Tc - T) ===
        plt.figure(figsize=(6, 4))
        plt.errorbar(log_T, log_M, yerr=log_M_errors, fmt='o', markersize=4,
                     color='darkorange', label='Simulated data', capsize=3)
        plt.plot(log_T, fit_line, 'r--', label=f'Fit (β ≈ {-beta_fit_line[0]:.3f})')
        
        plt.xlabel(r'$\log(T_c - T)$')
        plt.ylabel(r'$\log(M)$')
        plt.title(r'Log-log fit for $\beta$ (N = %d)' % Ni_int)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths['figures'] / f'loglog_beta_N{Ni_int}.png', dpi=300)
        plt.close()
        
        # === Graficar log(Cv) vs log(Tc - T) ===
        plt.figure(figsize=(6, 4))
        plt.errorbar(log_T, log_C, yerr=log_C_errors, fmt='o', markersize=4,
                     color='royalblue', label='Simulated data', capsize=3)
        plt.plot(log_T, fit_line_C, 'r--', label=f'Fit (α ≈ {-alpha_fit_line[0]:.3f})')
        
        plt.xlabel(r'$\log(T_c - T)$')
        plt.ylabel(r'$\log(C_v)$')
        plt.title(r'Log-log fit for $\alpha$ (N = %d)' % Ni_int)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths['figures'] / f'loglog_alpha_N{Ni_int}.png', dpi=300)
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
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'Tc_extrapolation.png', dpi=300)
plt.close()

# ===========================================
#   EXTRAPOLACIÓN DE β Y α AL LÍMITE N → ∞
# ===========================================
invN = 1.0 / N_values


# Extrapolar β
beta_inf, beta_err, beta_coef = extrapolate_exponent(N_values, beta_estimates, label='β')

plt.figure(figsize=(6, 4))
plt.errorbar(invN, beta_estimates, yerr=beta_errors, fmt='o', color='purple', label=r'$\beta(N)$', capsize=3)
plt.plot(invN, beta_coef[0]*invN + beta_coef[1],
         'r--', label=f'Fit: β = {beta_coef[0]:.2f}/N + {beta_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel(r'Critical exponent $\beta$')
plt.title(r'Extrapolation of $\beta$ to thermodynamic limit')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'beta_extrapolation.png', dpi=300)
plt.close()

# Extrapolar α
alpha_inf, alpha_err, alpha_coef = extrapolate_exponent(N_values, alpha_estimates, label='α')

plt.figure(figsize=(6, 4))
plt.errorbar(invN, alpha_estimates, yerr=alpha_errors, fmt='o', color='blue', label=r'$\alpha(N)$', capsize=3)
plt.plot(invN, alpha_coef[0]*invN + alpha_coef[1],
         'r--', label=f'Fit: α = {alpha_coef[0]:.2f}/N + {alpha_coef[1]:.3f}')
plt.xlabel('1/N')
plt.ylabel(r'Critical exponent $\alpha$')
plt.title(r'Extrapolation of $\alpha$ to thermodynamic limit')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.legend()
plt.savefig(paths['figures'] / 'alpha_extrapolation.png', dpi=300)
plt.close()


        
with open(
    paths['data'] / 'extrapolaciones_infinito.txt',
    'w', encoding='utf-8'
) as f:
    f.write(f"Tc(∞) = {Tc_inf:.6f} ± {Tc_err:.6f}\n")
    f.write(f"β(∞) = {beta_inf:.6f} ± {beta_err:.6f}\n")
    f.write(f"α(∞) = {alpha_inf:.6f} ± {alpha_err:.6f}\n")

# Final time testing
t_save_data_f = time.time()
t_save_gif_f = time.time()