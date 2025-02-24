# src/detectors/exponential_detector.py
from .base_detecor import BasePatternDetector
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from typing import Dict, Any

# If most of the exponential_var values are in a small range (e.g., 0 to 10),
# then the exponential pattern might not be well represented
# Can be considered as S curve

class ExponentialDetector(BasePatternDetector):
    def __init__(self, threshold = 0.3):
        super().__init__()
        self.threshold = threshold
    
    def exp_function(self, x, a, b):
        return a * np.exp(b * x)
    
    def detect(self, x, y):
        try:
            # Remove negative values
            mask = (x > 0) & (y > 0)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:  # Need at least 3 points
                return False
            
            # Fit exponential function
            popt, _ = curve_fit(self.exp_function, x_clean, y_clean)
            
            # Calculate R² score
            y_pred = self.exp_function(x_clean, *popt)
            # self.fit_score = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - y_clean.mean()) ** 2)
            self.fit_score = mean_squared_error(y_clean, y_pred)
            print("MSE of exponential fit:", self.fit_score)

            y_log = np.log(y_clean)
            y_pred_log = np.log(y_pred)
            r2_score = 1 - np.sum((y_log - y_pred_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
            print("Log Trasnformed R2 of exponential fit:", self.fit_score)

            

            self.params = {
                'a': popt[0],
                'b': popt[1]
            }
            # print(self.fit_score > self.threshold)
            return self.fit_score > self.threshold
            
        except:
            return False
    
    def get_parameters(self):
        return self.params
