from Base import PowerMartingale
from Dataset.heigh_weight import HeightWeightDataset
from Threshold import MartingaleThreshold
from p_values import PValueCalculator

from blackwater.estimation.distance.local_outlier_factor import LocalOutlierFactor

dataset = HeightWeightDataset()

model = LocalOutlierFactor(100, 10)
p_values = PValueCalculator()
threshold = MartingaleThreshold(false_positive_rate=0.1)
martingale = PowerMartingale(0.95, threshold.get_threshold(), p_values)


for sample in dataset:
    anomaly_score = model.score_one(sample)
    model.learn_one(sample)
    if anomaly_score is not None:
        p_values.update(anomaly_score)
        p_val = p_values.compute_p_value(anomaly_score)

        martingale_value = martingale.__call__(p_val)
