# auto-feat
automatic feature engineering

# System Architecture
![Alt text](s_curve_transform.png)

# Project Strucutre
src/
├── detectors/
│   ├── __init__.py
│   ├── base_detector.py
│   ├── s_curve_detector.py
│   └── exponential_detector.py
├── transformers/
│   ├── __init__.py
│   ├── base_transformer.py
│   └── s_curve_transformer.py
├── visualization/
│   ├── __init__.py
│   └── pattern_visualizer.py
└── main.py

Before and After Transformation:
![Alt text](s_curve_transform.png)
