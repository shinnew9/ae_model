import numpy as np
from scipy.stats import gamma

class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.threshold = None
        self.gamma_params = None

    def calculate_scores(self, data):
        reconstructed = self.model.predict(data)
        scores = np.mean(np.square(data - reconstructed), axis=1)
        return scores

    def fit_threshold(self, scores, percentile=0.9):
        self.gamma_params = gamma.fit(scores)
        self.threshold = gamma.ppf(percentile, *self.gamma_params)
        return self.threshold

    def detect_anomalies(self, scores):
        return scores > self.threshold