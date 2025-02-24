# src/detectors/linear_detector.py
from .base_detecor import BasePatternDetector
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Dict, Any

class LinearDetector(BasePatternDetector):
    def __init__(self, threshold = 0.6):
        super().__init__()
        self.threshold = threshold
    
    def detect(self, x, y):
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Predict y values using the fitted linear model
            y_pred = slope * x + intercept

            # Compute Mean Squared Error (MSE)
            self.fit_score = mean_squared_error(y, y_pred)

            # # r2_score
            # self.fit_score = r_value ** 2
            # print(self.fit_score)
            # print(p_value)
            # Store parameters
            
            self.params = {
                'slope': slope,
                'intercept': intercept,
                'p_value': p_value,
                'std_err': std_err
            }

            # Check if relationship is significantly linear
            is_significant = p_value < 0.05
            is_strong = abs(r_value) > self.threshold

            return is_significant and is_strong
        
        except:
            return False
    
    def get_parameters(self):
        return self.params
