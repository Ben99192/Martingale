import abc
import numpy as np
from scipy.stats import gaussian_kde


class BaseMartingale(abc.ABC):
    """Abstract base class for all Martingale implementations."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class PowerMartingale(BaseMartingale):
    """Stateful Callable Power Martingale."""

    def __init__(self, epsilon: float):
        self.epsilon: float = epsilon
        self.cumprod: float = 1.0  # Martingale starts at 1

    def __call__(self, p_val: float):
        """Updates the Martingale value based on the p-value."""
        self.cumprod *= self.epsilon * p_val ** (self.epsilon - 1)
        return self.cumprod


class MixtureMartingale(BaseMartingale):
    """Mixture Martingale combines multiple Power Martingales."""

    def __init__(self, epsilons=None, num_epsilons=100):
        if epsilons is None:
            self.epsilons = np.linspace(0.01, 1.0, num_epsilons)
        else:
            self.epsilons = epsilons

        self.martingales = [PowerMartingale(eps) for eps in self.epsilons]

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
    """Multi-View Martingale combines multiple feature-based Martingales."""

    def __init__(self, num_views: int, epsilon: float):
        self.num_views = num_views
        self.martingales = [PowerMartingale(epsilon) for _ in range(num_views)]

    def __call__(self, p_values: list):
        """Computes the maximum Martingale value across multiple feature views."""
        assert len(p_values) == self.num_views, "Incorrect number of p-values!"
        return max(m(p) for m, p in zip(self.martingales, p_values))


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

        self.cumprod *= rho_p  # Adjust Martingale value
        return self.cumprod
