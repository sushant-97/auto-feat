# src/detectors/base_detector.py
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional

class BasePatternDetector(ABC):
    """
    Base class for all pattern detectors.
    This should fit the current pattern on the given variables and return the parameters and respective score.
    """
    
    def __init__(self):
        self.params = {}
        self.fit_score = 0.0
        
    @abstractmethod
    def detect(self, x, y):
        """Detect if the pattern exists in the data."""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Return the parameters of the fitted pattern."""
        pass
    
    def get_fit_score(self):
        """Return the goodness of fit score."""
        return self.fit_score



