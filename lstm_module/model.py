"""
Dual-Head LSTM Architecture (Abdul Rehman)
==========================================
Two prediction heads sharing the same LSTM backbone:
  Head 1 (Rep Counter): Regression → predicts # of reps in window
  Head 2 (Form Classifier): Binary classification → good/bad form

LSTM INTERNALS (for viva):
  Each LSTM cell has 3 gates:
    - Forget gate: decides what info to DISCARD from cell state
    - Input gate:  decides what NEW info to STORE in cell state
    - Output gate: decides what to OUTPUT based on cell state
  Cell state = long-term memory highway (solves vanishing gradient)
  Hidden state = short-term working memory (used for predictions)
"""

import torch
import torch.nn as nn


class ExerciseLSTM(nn.Module):
    """
    Dual-head LSTM for exercise analysis.

    Architecture:
        Input (batch, seq_len, input_size)
            ↓
        LSTM Layer 1 (hidden=128, dropout=0.3)
            ↓
        LSTM Layer 2 (hidden=64, dropout=0.3)
            ↓
        ┌────────┴────────┐
        Rep Head          Form Head
        FC(64→32)→ReLU    FC(64→32)→ReLU
        FC(32→1)          FC(32→1)→Sigmoid
        (regression)      (classification)
    """

    def __init__(self, input_size=84, hidden_size_1=128, hidden_size_2=64,
                 dropout=0.3):
        """
        Args:
            input_size: Number of features per frame (default 84 from FeatureEngineer)
            hidden_size_1: Units in first LSTM layer
            hidden_size_2: Units in second LSTM layer
            dropout: Dropout rate between layers (prevents overfitting)
        """
        super(ExerciseLSTM, self).__init__()

        # LSTM Layer 1: processes input features → hidden representation
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,      # Input shape: (batch, seq, features)
            dropout=0,             # No dropout on single layer
        )

        # LSTM Layer 2: refines temporal patterns
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
            dropout=0,
        )

        # Dropout between LSTM and heads
        self.dropout = nn.Dropout(dropout)

        # Rep counting head (regression: predicts a number)
        self.rep_head = nn.Sequential(
            nn.Linear(hidden_size_2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),       # Output: single number (rep count)
        )

        # Form classification head (binary: good=1, bad=0)
        self.form_head = nn.Sequential(
            nn.Linear(hidden_size_2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),           # Output: probability of good form
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: (batch_size, seq_len, input_size) — sequence of frame features

        Returns:
            rep_count: (batch_size, 1) — predicted number of reps
            form_prob: (batch_size, 1) — probability of good form (0-1)
        """
        # LSTM Layer 1: capture low-level temporal patterns
        lstm1_out, _ = self.lstm1(x)  # (batch, seq, 128)

        # LSTM Layer 2: capture higher-level patterns
        lstm2_out, _ = self.lstm2(lstm1_out)  # (batch, seq, 64)

        # Use ONLY the last timestep's output for prediction
        # This summarizes the entire sequence into one vector
        last_hidden = lstm2_out[:, -1, :]  # (batch, 64)

        last_hidden = self.dropout(last_hidden)

        # Pass through both heads
        rep_count = self.rep_head(last_hidden)    # (batch, 1)
        form_prob = self.form_head(last_hidden)   # (batch, 1)

        return rep_count, form_prob


if __name__ == "__main__":
    # Quick test: verify shapes
    model = ExerciseLSTM(input_size=84)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fake batch: 4 samples, 60 frames each, 84 features per frame
    fake_input = torch.randn(4, 60, 84)
    rep_out, form_out = model(fake_input)
    print(f"Input shape:  {fake_input.shape}")
    print(f"Rep output:   {rep_out.shape} → {rep_out.detach().numpy().flatten()}")
    print(f"Form output:  {form_out.shape} → {form_out.detach().numpy().flatten()}")
    print("✅ LSTM model OK")
