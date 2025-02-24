# src/transformers/s_curve_transformer.py
from .base_transformer import BaseTransformer
import numpy as np

class SCurveTransformer(BaseTransformer):
    def __init__(self, params: dict):
        self.params = params
    
    def transform(self, x):
        """Apply logit transformation."""
        x_scaled = (x - x.min()) / (x.max() - x.min())
        return np.log(x_scaled / (1 - x_scaled))
    
    def get_name(self) -> str:
        return "Logit Transform"
