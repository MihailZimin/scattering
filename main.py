from test_functions import *

if __name__ == "__main__":
    # plot_scattering_cross_section_normalized_change_lambda()

    lmbda = 0.55 * 1e-6
    n_particle = 1.33 + 1j * 1e-8
    n_medium = 1
    radius = 0.26 * 1e-6

    # mie = Coefficients(n_medium, n_particle, lmbda, radius)
    # print(mie.x_)
    # plot_intensity_by_angle_theta(mie)
    # plot_scattering_cross_section_normalized_change_radius()
    # plot_scattering_cross_section_normalized_change_lambda()