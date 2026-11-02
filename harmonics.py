import scipy.special
import numpy as np

class SphericalHarmonics:
    def __init__(self, m: int, n: int, bessel: str = "j") -> None:
        """
        Calculates basis functions M_emn, M_omn, N_emn, N_omn for further usage.
        """

        self.m_ = m
        self.n_ = n
        self.bessel_ = bessel

    def bessel_normalized(self, ro: float, n: int = None) -> float:
        """
        Calculates bessel function.
        """
        if n is None:
            n = self.n_
            
        if self.bessel_ == "j":
            return np.sqrt(np.pi / (2 * ro)) * scipy.special.jv(n + 0.5, ro)
        elif self.bessel_ == "y":
            return np.sqrt(np.pi / (2 * ro)) * scipy.special.yv(n + 0.5, ro)

        raise ValueError(f"there is no such name for bessel function: {self.bessel_}")

    def derivative_legendre(self, theta):
        x = np.cos(theta)

        P, dP_dx = scipy.special.lpmn(self.m_, self.n_, x)
        dp_dx_val = dP_dx[self.m_, self.n_]

        dp_dtheta = -np.sin(theta) * dp_dx_val

        return dp_dtheta
    
    def derivative_bessel_normalized(self, ro: float) -> float:
        """
        Calculates d(z_n)/d(ro) using recurrence relation
        """
        n = self.n_
        zn_minus_1 = self.bessel_normalized(n - 1, ro)
        zn_plus_1 = self.bessel_normalized(n + 1, ro)
        return (n * zn_minus_1 - (n + 1) * zn_plus_1) / (2 * n + 1)

    def derivative_bessel_normalized_ro(self, ro):
        return self.bessel_normalized(ro) + ro * self.derivative_bessel_normalized(ro)
    
    def legendre(self, theta: float) -> float:
        return scipy.special.lpmv(self.m_, self.n_, np.cos(theta))
    
    def m_emn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        ro_part = 0
        theta_part = -self.m_ / np.sin(theta) * np.sin(self.m_ * fi) * self.legendre(theta) * self.bessel_normalized(ro)
        fi_part = -np.cos(self.m_ * fi) * self.derivative_legendre(theta) * self.bessel_normalized(ro)

        return np.array([ro_part, theta_part, fi_part])
    
    def m_omn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        ro_part = 0
        theta_part = self.m_ / np.sin(theta) * np.cos(self.m_ * fi) * self.legendre(theta) * self.bessel_normalized(ro)
        fi_part = -np.sin(self.m_ * fi) * self.derivative_legendre(theta) * self.bessel_normalized(ro)

        return np.array([ro_part, theta_part, fi_part])
    
    def n_emn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        ro_part = self.bessel_normalized(ro) / ro * np.cos(self.m_ * fi) * self.n_ * (self.n_ + 1) * self.legendre(theta)
        theta_part = np.cos(self.m_ * fi) * self.derivative_legendre(theta) / ro * self.derivative_bessel_normalized_ro(ro)
        fi_part = -self.m_ * np.sin(self.m_ * fi) * self.legendre(theta) / np.sin(theta) / ro * self.derivative_bessel_normalized_ro(ro)

        return np.array([ro_part, theta_part, fi_part])
    
    def n_omn(self, ro: float, fi: float, theta: float) -> np.ndarray[float]:
        ro_part = self.bessel_normalized(ro) / ro * np.sin(self.m_ * fi) * self.n_ * (self.n_ + 1) * self.legendre(theta)
        theta_part = np.sin(self.m_ * fi) * self.derivative_legendre(theta) / ro * self.derivative_bessel_normalized_ro(ro)
        fi_part = self.m_ * np.cos(self.m_ * fi) * self.legendre(theta) / np.sin(theta) / ro * self.derivative_bessel_normalized_ro(ro)

        return np.array([ro_part, theta_part, fi_part])
    
