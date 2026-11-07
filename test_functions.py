import numpy as np
import matplotlib.pyplot as plt
import refractiveindex as ri
from coefficients import *
from harmonics import *


def find_pi_and_tau_functions(n_max: int, theta: float):
    pi_funcs = np.zeros(n_max + 1)
    tau_funcs = np.zeros(n_max + 1)
    
    mu = np.cos(theta)

    pi_funcs[0] = 0
    pi_funcs[1] = 1
    
    for n in range(2, n_max + 1):
        pi_funcs[n] = ((2 * n - 1) / (n - 1)) * mu * pi_funcs[n - 1] - (n / (n - 1)) * pi_funcs[n - 2]
        
    n_arr = np.arange(n_max + 1)
    
    pi_minus_1 = np.roll(pi_funcs, 1) 
    pi_minus_1[0] = 0
    
    tau_funcs = n_arr * mu * pi_funcs - (n_arr + 1) * pi_minus_1
    return pi_funcs, tau_funcs


def plot_intensity_by_angle_theta(mie: Coefficients):
    coeffs_a, coeffs_b = mie.get_ab()
    nmax = mie.nmax_
    thetas = np.linspace(0, 180, 80)

    pi_vals = np.zeros(nmax)
    tau_vals = np.zeros(nmax)

    s1 = np.zeros_like(thetas)
    s2 = np.zeros_like(thetas)

    thetas_rad = np.radians(thetas)

    for i in range(thetas.shape[0]):
        pi_funcs, tau_funcs = find_pi_and_tau_functions(nmax, thetas_rad[i])

        n = np.arange(1, nmax + 1)
        coeff = (2 * n + 1) / n / (n + 1)
        s1[i] = np.sum(coeff * (coeffs_a * pi_funcs[1:] + coeffs_b * tau_funcs[1:]))
        s2[i] = np.sum(coeff * (coeffs_b * pi_funcs[1:] + coeffs_a * tau_funcs[1:]))

    fig, axes = plt.subplots(2, 1, figsize=(5, 8), subplot_kw={'projection': 'polar'}, layout='constrained')

    axes[0].plot(thetas_rad, np.absolute(s1) ** 2, color='royalblue', linewidth=2, label='$|S_1|^2$')
    axes[0].set_title('Перпендикулярная поляризация $|S_1|^2$', pad=15)
    axes[0].grid(True, linestyle='--')
    axes[0].set_thetalim(0, np.pi) 

    axes[1].plot(thetas_rad, np.absolute(s2) ** 2, color='crimson', linewidth=2, label='$|S_2|^2$')
    axes[1].set_title('Параллельная поляризация $|S_2|^2$', pad=15)
    axes[1].grid(True, linestyle='--')
    axes[1].set_thetalim(0, np.pi)
    plt.show()


def calculate_extinction_cross_section(mie: Coefficients):
    a_coeffs, b_coeffs = mie.get_ab()
    k = mie.k_

    n = np.arange(1, len(a_coeffs) + 1)
    n_array = 2 * n + 1

    coeff = 2 * np.pi / (k ** 2)

    return coeff * np.sum(n_array * np.real(a_coeffs + b_coeffs))


def calculate_scattering_cross_section(mie: Coefficients):
    a_coeffs, b_coeffs = mie.get_ab()
    k = mie.k_

    a_coeffs_abs_squared = np.absolute(a_coeffs) ** 2
    b_coeffs_abs_squared = np.absolute(b_coeffs) ** 2

    n = np.arange(1, len(a_coeffs) + 1)
    n_array = 2 * n + 1

    coeff = 2 * np.pi / (k ** 2)

    return coeff * np.sum(n_array * (a_coeffs_abs_squared + b_coeffs_abs_squared))


def plot_scattering_cross_section_normalized_change_radius():
    n_med = 1
    # n_part = 0.54 + 2.23j
    n_part = 1.33 + 0.0j
    lmbda = 532e-9  
    radius = 50e-9

    radiuses = np.linspace(radius, radius + 1e-5, 10000)

    x_vals = np.zeros(len(radiuses))
    c_sca = np.zeros(len(radiuses))

    for i, rad in enumerate(radiuses):
        mie = Coefficients(n_med, n_part, lmbda, rad)
        x_vals[i] = mie.x_
        c_sca[i] = calculate_scattering_cross_section(mie)

    q_sca = c_sca / np.pi / (radiuses ** 2)
    indmax = np.argmax(q_sca)
    r_max = radiuses[indmax]
    print(r_max)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, q_sca, label='$C_{sca}$', color='crimson')
    plt.xlabel('Параметр размера $x = 2\pi n_{med} a / \lambda_0$')
    plt.ylabel('Эффективность рассеяния $Q_{sca}$')
    plt.grid(True, linestyle='--')
    # plt.yscale('log')
    plt.title('Зависимость эффективности рассеяния Ми от размера частицы')
    plt.savefig('results/scattering_efficiency_radius_var.png')
    plt.show()


def plot_scattering_cross_section_normalized_change_lambda():
    water = ri.RefractiveIndexMaterial(shelf='main', book='H2O', page='Segelstein')
    # water = ri.RefractiveIndexMaterial(shelf='main', book='H2O', page='Hale')
    wave_steps = np.linspace(100, 10000, 100000)
    refraction_coeffs = water.get_refractive_index(wave_steps)
    extinction_coeffs = water.get_extinction_coefficient(wave_steps)

    n_vals = refraction_coeffs + 1j * extinction_coeffs
    n_medium = 1.0 + 0.0j
    radius = 1*1e-6

    x_vals = np.zeros_like(n_vals)
    c_sca = np.zeros_like(n_vals)

    for i, n_particle in enumerate(n_vals):
        mie = Coefficients(n_medium, n_particle, wave_steps[i] * 1e-9, radius)
        x_vals[i] = mie.x_
        c_sca[i] = calculate_scattering_cross_section(mie)

    q_sca = c_sca / np.pi / (radius ** 2)
    
    plt.figure(figsize=(8, 5))
    plt.plot(1/(wave_steps * 0.001), q_sca, label='$Q_{sca}$', color='crimson')
    # plt.xlabel('Параметр размера $x = 2\pi n_{med} a / \lambda$')
    plt.xlabel('Параметр $1 / \lambda$')
    plt.ylabel('Эффективность рассеяния $Q_{sca}$')
    plt.grid(True, linestyle='--')
    # plt.yscale('log')
    plt.title('Зависимость эффективности рассеяния Ми от длины падающей волны на водяную каплю')
    plt.savefig('results/scattering_efficiency_lambda_var.png')
    plt.show()



def plot_extinction_cross_section_normalized_change_lambda():
    water = ri.RefractiveIndexMaterial(shelf='main', book='H2O', page='Segelstein')
    # water = ri.RefractiveIndexMaterial(shelf='main', book='H2O', page='Hale')
    wave_steps = np.linspace(100, 100000, 500000)
    refraction_coeffs = water.get_refractive_index(wave_steps)
    extinction_coeffs = water.get_extinction_coefficient(wave_steps)

    n_vals = refraction_coeffs + 1j * extinction_coeffs
    n_medium = 1.0 + 0.0j
    radius = 1*1e-6

    x_vals = np.zeros_like(n_vals)
    c_ext = np.zeros_like(n_vals)

    for i, n_particle in enumerate(n_vals):
        mie = Coefficients(n_medium, n_particle, wave_steps[i] * 1e-9, radius)
        x_vals[i] = mie.x_
        c_ext[i] = calculate_extinction_cross_section(mie)

    q_ext = c_ext / np.pi / (radius ** 2)
    
    plt.figure(figsize=(8, 5))
    plt.plot(1/(wave_steps * 0.001), q_ext, label='$Q_{ext}$', color='crimson')
    # plt.xlabel('Параметр размера $x = 2\pi n_{med} a / \lambda$')
    plt.xlabel('Параметр $1 / \lambda$')
    plt.ylabel('Эффективность экстинкции $Q_{ext}$')
    plt.grid(True, linestyle='--')
    # plt.yscale('log')
    plt.title('Зависимость эффективности экстинкции Ми от длины падающей волны на водяную каплю')
    plt.savefig('results/extinction_efficiency_lambda_var.png')
    plt.show()