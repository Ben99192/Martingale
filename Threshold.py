class MartingaleThreshold:
    """ Determines the Martingale threshold based on a desired False Positive Rate. """

    def __init__(self, false_positive_rate: float = 0.05):
        assert 0 < false_positive_rate < 1, "FPR must be between 0 and 1!"
        self.alpha = false_positive_rate
        self.lambda_value = 1 / self.alpha  # Compute λ

    def check_martingale(self, martingale_value: float) -> bool:
        """ Checks if the Martingale exceeds the threshold λ. """
        return martingale_value > self.lambda_value

    def update_threshold(self, new_false_positive_rate: float):
        """ Updates the threshold based on a new False Positive Rate. """
        assert 0 < new_false_positive_rate < 1, "FPR must be between 0 and 1!"
        self.alpha = new_false_positive_rate
        self.lambda_value = 1 / self.alpha  # Recompute λ

    def get_threshold(self) -> float:
        """ Returns the current λ threshold value. """
        return self.lambda_value

