# auto-feat
automatic feature engineering

# System Architecture: Feature Pattern Analysis and Transformation

## 1. Project Strucutre
```
src/
├── __init__.py
├── main.py               # Main class and orchestration
├── detectors/           # Pattern detection modules
│   ├── __init__.py
│   ├── base.py         # Base detector class
│   ├── linear_detector.py       # Linear pattern detector
│   ├── exponential_detector.py  # Exponential pattern detector
│   ├── s_curve_detector.py      # S-curve pattern detector
│   └── decay_detector.py       # Decay pattern detector
├── transformers/       # Data transformation modules
│   ├── __init__.py
│   ├── base.py        # Base transformer class
│   ├── log.py         # Log transformer
│   ├── logit.py       # Logit transformer/ s_curve
│   └── boxcox.py      # BoxCox transformer
└── visualization/     # Visualization modules
    ├── __init__.py
    └── pattern_visualizer.py      # Plotting utilities
```

## 2. Core Components

```python
# detectors/base.py
class BaseDetector:
    def __init__(self):
        self.params = {}
        self.fit_score = 0.0
        
    @abstractmethod
    def detect(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Detect if pattern exists in data"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict:
        """Return pattern parameters"""
        pass
    
    def get_fit_score(self) -> float:
        """Return goodness of fit"""
        return self.fit_score

# transformers/base.py
class BaseTransformer:
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseTransformer':
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
```

## 3. Before and After Transformation:
![Alt text](s_curve_transform.png)
