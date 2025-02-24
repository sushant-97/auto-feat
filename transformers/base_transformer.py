# src/transformers/base_transformer.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseTransformer(ABC):
    """Base class for all feature transformers."""
    
    @abstractmethod
    def fit(self, X):
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform the input feature."""
        pass
    
    def fit_transform(self, X):
        """Fit the transformer and transform the data."""
        return self.fit(X).transform(X)

class LogTransformer:
    def fit(self, X):
        self.min_val = np.min(X)
        return self
    
    def transform(self, X):
        return np.log1p(X - self.min_val + 1)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LogitTransformer:
    def fit(self, X):
        self.min_val = np.min(X)
        self.max_val = np.max(X)
        return self
    
    def transform(self, X):
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val)
        X_scaled = np.clip(X_scaled, 0.001, 0.999)
        return np.log(X_scaled / (1 - X_scaled))
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class IdentityTransformer:
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return X
