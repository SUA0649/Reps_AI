"""
LSTM Module — Temporal Sequence Modeling Pipeline (Abdul Rehman)
================================================================
Handles auto-labeling, dataset creation, LSTM architecture,
and training for rep counting + form classification.
"""

from .model import ExerciseLSTM
from .auto_labeler import AutoLabeler
from .dataset import ExerciseDataset
