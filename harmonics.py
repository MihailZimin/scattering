import numpy as np
import scipy.special

class SphericalHarmonics:
    def __init__(self, m: int, n: int, bessel: str = "j") -> None:
        """
        Рассчитывает векторные сферические гармоники M_emn, M_omn, N_emn, N_omn.
        """
        self.m_ = m
        self.n_ = n
        self.bessel_ = bessel

    def _bessel(self, order: int, ro: np.ndarray) -> np.ndarray:
        if self.bessel_ == "j":
            return scipy.special.spherical_jn(order, ro)
        elif self.bessel_ == "y":
            return scipy.special.spherical_yn(order, ro)
        raise ValueError(f"Неизвестный тип функции Бесселя: {self.bessel_}")
    
    def _bessel_derivative(self, order: int, ro: np.ndarray) -> np.ndarray:
        if self.bessel_ == "j":
            return scipy.special.spherical_jn(order, ro, derivative=True)
        elif self.bessel_ == "y":
            return scipy.special.spherical_yn(order, ro, derivative=True)
        raise ValueError(f"Неизвестный тип функции Бесселя: {self.bessel_}")
    
    def bessel_normalized(self, ro: np.ndarray) -> np.ndarray:
        return self._bessel(self.n_, ro)
    
    def derivative_bessel_normalized_ro(self, ro: np.ndarray) -> np.ndarray:
        """Вычисляет d(z_n(ro) * ro) / d(ro) = z_n(ro) + ro * z_n'(ro)"""
        return self._bessel(self.n_, ro) + ro * self._bessel_derivative(self.n_, ro)
    
    def angular_functions(self, theta: np.ndarray, fi: np.ndarray):
        """
        Вычисляет угловые зависимости, обходя сингулярности на полюсах (sin(theta)=0).
        Возвращает:
          pi_mn = m * P_n^m / sin(theta)
          tau_mn = d(P_n^m) / d(theta)
        """
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        p_n = scipy.special.lpmv(self.m_, self.n_, cos_t)
        p_nm1 = scipy.special.lpmv(self.m_, self.n_ - 1, cos_t) if self.n_ > 0 else np.zeros_like(p_n)
    
        with np.errstate(divide='ignore', invalid='ignore'):
            pi_mn = self.m_ * p_n / sin_t
            tau_mn = (self.n_ * cos_t * p_n - (self.n_ + self.m_) * p_nm1) / sin_t
        
        pi_mn = np.nan_to_num(pi_mn)
        tau_mn = np.nan_to_num(tau_mn)
        
        sin_mf = np.sin(self.m_ * fi)
        cos_mf = np.cos(self.m_ * fi)
        
        return p_n, pi_mn, tau_mn, sin_mf, cos_mf
    
    def m_emn(self, ro: np.ndarray, fi: np.ndarray, theta: np.ndarray) -> np.ndarray[float]:
        """
        Вычисляет гармонику через формулу функций, зависящих от угла.
        """
        p_n, pi_mn, tau_mn, sin_mf, cos_mf = self.angular_functions(theta, fi)
        z_n = self.bessel_normalized(ro)

        ro_part = np.zeros_like(z_n)
        theta_part = -sin_mf * pi_mn * z_n
        fi_part = -cos_mf * tau_mn * z_n
        return np.stack([ro_part, theta_part, fi_part], axis=0)
    
    def m_omn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        p_n, pi_mn, tau_mn, sin_mf, cos_mf = self.angular_functions(theta, fi)
        z_n = self.bessel_normalized(ro)

        ro_part = np.zeros_like(z_n)
        theta_part = cos_mf * pi_mn * z_n
        fi_part = -sin_mf * tau_mn * z_n

        return np.stack([ro_part, theta_part, fi_part], axis=0)
    
    def n_emn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        p_n, pi_mn, tau_mn, sin_mf, cos_mf = self.angular_functions(theta, fi)
        z_n = self.bessel_normalized(ro)
        n = self.n_
        sin_theta = np.sin(theta)

        ro_part = cos_mf * n * (n + 1) * sin_theta * pi_mn * z_n / ro
        theta_part = cos_mf * tau_mn * self.derivative_bessel_normalized_ro(ro) / ro
        fi_part = -sin_mf * pi_mn * self.derivative_bessel_normalized_ro(ro) / ro

        return np.stack([ro_part, theta_part, fi_part], axis=0)
    

    def n_omn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        p_n, pi_mn, tau_mn, sin_mf, cos_mf = self.angular_functions(theta, fi)
        z_n = self.bessel_normalized(ro)
        n = self.n_
        sin_theta = np.sin(theta)

        ro_part = sin_mf * n * (n + 1) * sin_theta * pi_mn * z_n / ro
        theta_part = sin_mf * tau_mn * self.derivative_bessel_normalized_ro(ro) / ro
        fi_part = cos_mf * pi_mn * self.derivative_bessel_normalized_ro(ro) / ro

        return np.stack([ro_part, theta_part, fi_part], axis=0)
    
