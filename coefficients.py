import numpy as np
import scipy.special

class Coefficients:
    def __init__(self, n_medium: complex, n_particle: complex, lmbda: float, a: float):
        """
        Инициализирует класс вычисления коэффициентов

        n_medium - показатель преломления среды
        n_particle - показатель преломления частицы
        lmbda - длина волны
        a - радиус частицы
        """

        self.n_medium_ = n_medium
        self.n_particle_ = n_particle
        self.lambda_ = lmbda
        self.a_ = a
        self.m_ = n_particle / n_medium
        self.k_ = 2 * np.pi * n_medium / lmbda
        self.x_ = self.k_ * a
        self.nmax_ = 1

        x = self.x_.real
        if x <= 8:
            self.nmax_ = int(np.ceil(x + 4 * x ** (1/3) + 1))
        elif x <= 4200:
            self.nmax_ = int(np.ceil(x + 4.05 * x ** (1/3) + 2))
        else:
            self.nmax_ = int(np.ceil(x + 4 * x ** (1/3) + 2))

    def psi_ro_range(self, ro: complex):
        """
        Вычисляет все psi в диапазоне от 0 до self.nmax_
        """
    
        n_row = np.arange(0, self.nmax_ + 1)
        return ro * scipy.special.spherical_jn(n_row, ro)
    
    def psi_ro_derivative_range(self, ro: complex):
    
        n_row = np.arange(0, self.nmax_ + 1)
        return self.psi_ro_range(ro) + ro * scipy.special.spherical_jn(n_row, ro, derivative=True)
    
    def _spherical_hankel1_range(self, ro: complex):
        """Сферическая функция Ханкеля 1-го рода порядка n от аргумента z."""
    
        n_row = np.arange(0, self.nmax_ + 1)
        return scipy.special.spherical_jn(n_row, ro) + 1j * scipy.special.spherical_yn(n_row, ro)
    
    def _spherical_hankel1_derivative(self, ro: complex):
    
        n_row = np.arange(0, self.nmax_ + 1)
        return scipy.special.spherical_jn(n_row, ro, derivative=True) + 1j * scipy.special.spherical_yn(n_row, ro, derivative=True)
    
    def ksi_ro_range(self, ro: np.ndarray):

        return ro * self._spherical_hankel1_range(ro)
    
    def ksi_ro_derivative(self, ro: np.ndarray):
        return self._spherical_hankel1_range(ro) + ro * self._spherical_hankel1_derivative(ro)
    
    def D_ro_range(self, ro: complex):
        n_iters = self.nmax_ + 20
        
        d_range = np.zeros(n_iters, dtype=complex)

        for i in range(n_iters - 2, -1, -1):
            d_range[i] = (i + 1) / ro - 1 / (d_range[i + 1] + (i + 1) / ro)

        return d_range[0:self.nmax_ + 1].copy()
    
    def coefficient_a(self):
        n = np.arange(1, self.nmax_ + 1)
        x = self.x_
        m = self.m_
        psi_ro = self.psi_ro_range(x)
        D_ro = self.D_ro_range(m * x)
        ksi_ro = self.ksi_ro_range(x)

        nom = (D_ro[1:] / m + n / x) * psi_ro[1:] - psi_ro[0:-1]
        denom = (D_ro[1:] / m + n / x) * ksi_ro[1:] - ksi_ro[0:-1]

        return nom / denom
    
    def coefficient_b(self):
        n = np.arange(1, self.nmax_ + 1)
        x = self.x_
        m = self.m_
        psi_ro = self.psi_ro_range(x)
        D_ro = self.D_ro_range(m * x)
        ksi_ro = self.ksi_ro_range(x)

        nom = (D_ro[1:] * m + n / x) * psi_ro[1:] - psi_ro[0:-1]
        denom = (D_ro[1:] * m + n / x) * ksi_ro[1:] - ksi_ro[0:-1]

        return nom / denom
    
    def get_ab(self):
        n = np.arange(1, self.nmax_ + 1)
        psi = self.psi_ro_range(self.x_)
        D = self.D_ro_range(self.m_ * self.x_)
        ksi = self.ksi_ro_range(self.x_)
        a = ((D[1:] / self.m_ + n / self.x_) * psi[1:] - psi[0:-1]) / ((D[1:] / self.m_ + n / self.x_) * ksi[1:] - ksi[0:-1])
        b = ((D[1:] * self.m_ + n / self.x_) * psi[1:] - psi[0:-1]) / ((D[1:] * self.m_ + n / self.x_) * ksi[1:] - ksi[0:-1])
        return a, b