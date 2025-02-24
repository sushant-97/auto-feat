# src/detectors/decay_detector.py
from .base_detecor import BasePatternDetector
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Any

class DecayDetector(BasePatternDetector):
    def __init__(self, threshold = 0.6):
        super().__init__()
        self.threshold = threshold
    
    def decay_function(self, x, a, b, c):
        """Exponential decay function: f(x) = a * exp(-b * x) + c"""
        return a * np.exp(-b * x) + c
    
    def detect(self, x,y):
        try:
            # Initial parameter guess
            a_guess = np.max(y) - np.min(y)
            b_guess = 1.0
            c_guess = np.min(y)
            
            # Fit decay function
            popt, pcov = curve_fit(
                self.decay_function, 
                x, 
                y,
                p0=[a_guess, b_guess, c_guess],
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])
            )
            
            # Calculate R² score
            y_pred = self.decay_function(x, *popt)
            self.fit_score = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            
            # Store parameters
            self.params = {
                'amplitude': popt[0],
                'decay_rate': popt[1],
                'offset': popt[2],
                'half_life': np.log(2) / popt[1] if popt[1] > 0 else None
            }
            
            return self.fit_score > self.threshold
            
        except:
            return False
    
    def get_parameters(self):
        return self.params
