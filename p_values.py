import numpy as np


class PValueCalculator:
    """Computes p-values from stored strangeness values."""

    def __init__(self):
        self.T = []  # Speicher für vorherige Strangeness-Werte

    def update(self, new_strangeness: float):
        """Stores a new strangeness value."""
        self.T.append(new_strangeness)

    def compute_p_value(self, new_strangeness: float):
        """Computes p-value for the given strangeness value."""
        if not self.T:
            return 1.0  # Default to 1 if no previous values exist

        i = len(self.T)
        count_greater = sum(1 for s in self.T if s > new_strangeness)
        count_equal = sum(1 for s in self.T if s == new_strangeness)
        xi = np.random.uniform(0, 1)  # Random tie-breaking factor

        return (count_greater + xi * count_equal) / i

    def reset_T(self):
        """Resets T"""
        print("🔄 Resetting T in PValueCalculator")
        self.T = []
