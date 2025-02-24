# src/detectors/s_curve_detector.py
from scipy.optimize import curve_fit
from .base_detecor import BasePatternDetector
import numpy as np
from typing import Dict, Any

class SCurveDetector(BasePatternDetector):
    def __init__(self, threshold= 0.8):
        super().__init__()
        self.threshold = threshold
        
    def logistic_function(self, x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    def detect(self, x, y):
        try:
            # Normalize data
            x_norm = (x - x.min()) / (x.max() - x.min())
            y_norm = (y - y.min()) / (y.max() - y.min())
            
            # Fit logistic function
            popt, _ = curve_fit(self.logistic_function, x_norm, y_norm, 
                              p0=[1.0, 1.0, 0.5], bounds=([0, 0, 0], [2, 20, 1]))
            
            # Calculate R² score
            y_pred = self.logistic_function(x_norm, *popt)
            self.fit_score = 1 - np.sum((y_norm - y_pred) ** 2) / np.sum((y_norm - y_norm.mean()) ** 2)
            print("S curve fit score:", self.fit_score)
            # Store parameters
            self.params = {
                'L': popt[0] * (y.max() - y.min()) + y.min(),
                'k': popt[1],
                'x0': popt[2] * (x.max() - x.min()) + x.min()
            }
            
            return self.fit_score > self.threshold
            
        except:
            return False
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.params
