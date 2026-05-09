"""
LSTM Inference Module (Abdul Rehman)
=====================================
Loads a trained ExerciseLSTM and runs it on a single feature sequence.
Used by the real-time system and the evaluation script.

Usage:
  from lstm_module.inference import LSTMInference

  inferencer = LSTMInference('models/lstm_best.pth')
  rep_count, form_prob = inferencer.predict(feature_sequence)  # (60, 84) array
"""

import torch
import numpy as np
from pathlib import Path

from lstm_module.model import ExerciseLSTM


class LSTMInference:
    """
    Wrapper around ExerciseLSTM for clean single-sequence inference.

    INFERENCE CONCEPTS (for viva):
      - model.eval(): disables dropout (we want deterministic output at test time)
      - torch.no_grad(): skips gradient computation (faster, less memory)
      - We trained with window_size=60 — inference must use the same window size
    """

    def __init__(self, model_path='models/lstm_best.pth', input_size=84, device=None):
        """
        Load a trained LSTM model from a .pth weights file.

        Args:
            model_path: path to the saved model weights (from train.py)
            input_size: must match training config (default 84)
            device: 'cuda' or 'cpu' — auto-detected if None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train the model first with: python -m lstm_module.train"
            )

        # Load model architecture and weights
        self.model = ExerciseLSTM(input_size=input_size).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()  # IMPORTANT: switch to eval mode (disables dropout)

        print(f"✅ LSTM loaded from {model_path} on {self.device}")

    def predict(self, feature_sequence):
        """
        Run inference on a single feature sequence.

        Args:
            feature_sequence: numpy array of shape (seq_len, 84)
                              seq_len should be 60 (window_size used during training)

        Returns:
            rep_count: float — predicted number of reps in the window
            form_prob: float — probability of good form (0.0 = bad, 1.0 = good)
        """
        if not isinstance(feature_sequence, np.ndarray):
            feature_sequence = np.array(feature_sequence)

        if feature_sequence.ndim != 2 or feature_sequence.shape[1] != self.input_size:
            raise ValueError(
                f"Expected shape (seq_len, {self.input_size}), "
                f"got {feature_sequence.shape}"
            )

        # Add batch dimension: (seq_len, 84) → (1, seq_len, 84)
        x = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rep_pred, form_pred = self.model(x)

        rep_count = float(rep_pred.item())
        form_prob = float(form_pred.item())

        return rep_count, form_prob

    def predict_batch(self, sequences):
        """
        Run inference on multiple sequences at once (faster than one by one).

        Args:
            sequences: list of numpy arrays, each (seq_len, 84)
                       OR numpy array of shape (batch_size, seq_len, 84)

        Returns:
            rep_counts: numpy array of shape (batch_size,)
            form_probs: numpy array of shape (batch_size,)
        """
        if isinstance(sequences, list):
            sequences = np.stack(sequences)  # (batch, seq_len, 84)

        x = torch.FloatTensor(sequences).to(self.device)

        with torch.no_grad():
            rep_preds, form_preds = self.model(x)

        return rep_preds.cpu().numpy().flatten(), form_preds.cpu().numpy().flatten()

    def interpret(self, rep_count, form_prob):
        """
        Convert raw model outputs to human-readable strings.

        Args:
            rep_count: float from predict()
            form_prob: float from predict()

        Returns:
            dict with 'reps', 'form_label', 'form_color' (for overlay display)
        """
        rounded_reps = round(rep_count)
        rounded_reps = max(0, rounded_reps)  # clamp to non-negative

        if form_prob > 0.7:
            form_label = "GOOD FORM"
            form_color = (0, 200, 0)      # Green (BGR for OpenCV)
        elif form_prob > 0.4:
            form_label = "CHECK FORM"
            form_color = (0, 200, 200)    # Yellow
        else:
            form_label = "BAD FORM"
            form_color = (0, 0, 200)      # Red

        return {
            'reps': rounded_reps,
            'rep_raw': rep_count,
            'form_label': form_label,
            'form_prob': form_prob,
            'form_color': form_color,
        }


if __name__ == "__main__":
    # Quick test: load model and run on fake data
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/lstm_best.pth'

    inferencer = LSTMInference(model_path)

    # Fake input: 60 frames, 84 features each (simulate 2 seconds of squatting)
    fake_sequence = np.random.randn(60, 84).astype(np.float32)

    rep_count, form_prob = inferencer.predict(fake_sequence)
    result = inferencer.interpret(rep_count, form_prob)

    print(f"\nTest inference result:")
    print(f"  Rep count (raw):  {rep_count:.2f}")
    print(f"  Rep count (int):  {result['reps']}")
    print(f"  Form probability: {form_prob:.2f}")
    print(f"  Form label:       {result['form_label']}")
    print("✅ Inference module OK")
