import abc
import numpy as np
from scipy.stats import gaussian_kde
from p_values import PValueCalculator


class BaseMartingale(abc.ABC):
    """Abstract base class for all Martingale implementations."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PowerMartingale(BaseMartingale):
    """Power Martingale with reset functionality after threshold exceedance."""

    def __init__(
        self, epsilon: float, threshold: float, p_value_calculator: PValueCalculator
    ):
        self.epsilon: float = epsilon
        self.threshold: float = threshold
        self.cumprod: float = 1.0
        self.p_value_calculator = p_value_calculator

    def __call__(self, p_val: float):
        """Updates the Martingale value and resets it if it exceeds the threshold."""
        self.cumprod *= self.epsilon * max(p_val, 1e-10) ** (self.epsilon - 1)

        if self.cumprod > self.threshold:  # If threshold exceeded, reset
            print("🚨 Anomaly detected! Resetting Martingale.")
            print(self.cumprod)
            self.cumprod = 1.0  # Reset Martingale
            self.p_value_calculator.reset_T()

        return self.cumprod


class MixtureMartingale(BaseMartingale):
    """Mixture Martingale combines multiple Power Martingales."""

    def __init__(self, epsilons=None, num_epsilons=100, threshold=10):
        if epsilons is None:
            self.epsilons = np.linspace(0.01, 1.0, num_epsilons)
        else:
            self.epsilons = epsilons

        self.martingales = [PowerMartingale(eps, threshold) for eps in self.epsilons]

    def __call__(self, p_val: float):
        """Computes the mixture Martingale by averaging multiple Power Martingales."""
        return np.mean([m(p_val) for m in self.martingales])


class SPRTMartingale(BaseMartingale):
    """Sequential Probability Ratio Test (SPRT) using Martingale approach."""

    def __init__(self, epsilon: float, threshold: float):
        self.epsilon = epsilon
        self.threshold = threshold
        self.martingale = PowerMartingale(epsilon)

    def __call__(self, p_val: float):
        """Checks if the Martingale exceeds the threshold."""
        value = self.martingale(p_val)
        if value > self.threshold:
            print("🚨 Change detected!")
        return value


class MultiViewMartingale(BaseMartingale):
    """Multi-View Martingale resets if any view exceeds the threshold."""

    def __init__(self, num_views: int, epsilon: float, threshold: float):
        self.num_views = num_views
        self.threshold = threshold
        self.martingales = [
            PowerMartingale(epsilon, threshold) for _ in range(num_views)
        ]

    def __call__(self, p_values: list):
        """Computes the max Martingale across views and resets if needed."""

        max_martingale = max(m(p) for m, p in zip(self.martingales, p_values))

        if max_martingale > self.threshold:
            print("🚨 Anomaly detected across multiple views! Resetting Martingales.")
            for m in self.martingales:
                m.cumprod = 1.0  # Reset all Martingales in Multi-View setting

        return max_martingale


class PluginMartingale(BaseMartingale):
    """Plugin Martingale with density estimation."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.cumprod = 1.0
        self.p_values = []

    def __call__(self, p_val: float):
        """Updates the Martingale using estimated density from KDE."""
        self.p_values.append(p_val)

        if len(self.p_values) < 5:  # Minimum required points for KDE
            return self.cumprod

        kde = gaussian_kde(self.p_values)
        rho_p = kde.evaluate(p_val)[0]  # Density estimate at p_val

        self.cumprod *= self.epsilon * rho_p
        return self.cumprod
