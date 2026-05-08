"""
CNN Module — Spatial Feature Extraction Pipeline (Shaheer)
==========================================================
Handles video preprocessing, keypoint extraction via MediaPipe,
and feature engineering (joint angles, normalization, velocities).
"""

from .keypoint_extractor import KeypointExtractor
from .feature_engineer import FeatureEngineer
