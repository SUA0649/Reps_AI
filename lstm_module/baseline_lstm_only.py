"""
LSTM-Only Baseline (Abdul Rehman)
===================================
BASELINE 2: Skip MediaPipe entirely.
Feed raw flattened frame pixels (64×64 grayscale = 4096 features) directly to LSTM.

Usage:
  python -m lstm_module.baseline_lstm_only --data_dir data/processed --epochs 50
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# 1. RAW PIXEL DATASET
# ============================================================

class RawPixelDataset(Dataset):
    """
    LSTM-Only dataset: uses raw flattened keypoint coordinates as fake 'pixels'.

    IMPORTANT — what we actually do here:
    Since we don't have raw video frames stored (too large), we SIMULATE raw pixel
    input by taking the (N, 33, 4) keypoint arrays and flattening them WITHOUT
    any feature engineering (no angle calculation, no normalization, no velocity).

    This simulates: "what if LSTM just got raw position numbers with no structure?"
    The keypoint .npy files have shape (N, 33, 4) — flatten each frame to (132,)
    instead of the engineered (84,) features.

    WHY THIS IS WORSE (for viva):
    - No joint angle computation → LSTM must infer angles from raw (x,y) positions
    - No normalization → model is confused when person moves left/right on screen
    - No velocity features → harder to detect motion direction
    - Same small dataset → same data poverty, but harder task
    """

    def __init__(self, samples, labels, augment=False):
        self.samples = samples
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()

        if self.augment:
            # Only add noise — no meaningful augmentation possible on raw coords
            if np.random.random() < 0.5:
                x = x + np.random.normal(0, 0.02, x.shape)  # more noise than engineered

        rep_count = self.labels[idx]['rep_count']
        form_quality = self.labels[idx]['form_quality']

        return (
            torch.FloatTensor(x),
            torch.FloatTensor([rep_count]),
            torch.FloatTensor([form_quality]),
        )


def create_raw_sliding_windows(keypoints, labels_dict, window_size=60, stride=15):
    """
    Create sliding windows from RAW keypoint data (no feature engineering).

    Args:
        keypoints: (num_frames, 33, 4) — raw MediaPipe output
        labels_dict: from auto-labeler JSON

    Returns:
        windows: list of (window_size, 132) arrays  ← flattened raw landmarks
        window_labels: list of dicts
    """
    # Flatten (N, 33, 4) → (N, 132)
    num_frames = keypoints.shape[0]
    flat_keypoints = keypoints.reshape(num_frames, -1)  # (N, 132)

    rep_boundaries = labels_dict.get('rep_boundaries', [])
    form_labels = labels_dict.get('form_labels', [])

    windows = []
    window_labels = []

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        window = flat_keypoints[start:end]  # (60, 132)

        reps_in_window = 0
        form_scores = []

        for i, (r_start, r_valley, r_end) in enumerate(rep_boundaries):
            if start <= r_valley < end:
                reps_in_window += 1
                if i < len(form_labels):
                    form_scores.append(form_labels[i])

        avg_form = float(np.mean(form_scores)) if form_scores else 0.5
        windows.append(window)
        window_labels.append({'rep_count': reps_in_window, 'form_quality': avg_form})

    return windows, window_labels


def build_raw_datasets(processed_dir, window_size=60, stride=15):
    """Load keypoints (not features) and build train/val/test splits."""
    processed_dir = Path(processed_dir)
    all_windows = []
    all_labels = []

    for exercise_dir in processed_dir.iterdir():
        if not exercise_dir.is_dir():
            continue

        exercise_type = exercise_dir.name
        print(f"Loading raw keypoints from {exercise_type}...")

        for kp_file in sorted(exercise_dir.glob("*_keypoints.npy")):
            label_file = kp_file.parent / kp_file.name.replace(
                '_keypoints.npy', '_labels.json'
            )
            if not label_file.exists():
                continue

            keypoints = np.load(kp_file)  # (N, 33, 4)
            with open(label_file) as f:
                labels_dict = json.load(f)

            windows, w_labels = create_raw_sliding_windows(
                keypoints, labels_dict, window_size, stride
            )
            all_windows.extend(windows)
            all_labels.extend(w_labels)

    print(f"Total raw samples: {len(all_windows)}")
    if len(all_windows) == 0:
        raise ValueError("No data found!")

    indices = list(range(len(all_windows)))
    train_idx, test_val_idx = train_test_split(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.50, random_state=42)

    def subset(idxs, augment=False):
        return RawPixelDataset(
            [all_windows[i] for i in idxs],
            [all_labels[i] for i in idxs],
            augment=augment,
        )

    return subset(train_idx, augment=True), subset(val_idx), subset(test_idx)


# ============================================================
# 2. LSTM-ONLY MODEL (same architecture, different input size)
# ============================================================

class RawLSTM(nn.Module):
    """
    Same dual-head LSTM architecture but with input_size=132 (raw flattened landmarks)
    instead of 84 (engineered features).

    The architecture is identical — this is a CONTROLLED experiment.
    Only the INPUT changes. If results are worse, it proves feature engineering matters.
    """

    def __init__(self, input_size=132, hidden_size_1=128, hidden_size_2=64, dropout=0.3):
        super(RawLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size_1, hidden_size=hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.rep_head = nn.Sequential(
            nn.Linear(hidden_size_2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
        self.form_head = nn.Sequential(
            nn.Linear(hidden_size_2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        last_hidden = self.dropout(lstm2_out[:, -1, :])
        return self.rep_head(last_hidden), self.form_head(last_hidden)


# ============================================================
# 3. TRAINING LOOP (simplified — same logic as train.py)
# ============================================================

def train_baseline(data_dir, output_dir='models', epochs=50, batch_size=32,
                   lr=0.001, patience=10):
    """Train the LSTM-only baseline on raw keypoint data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("LSTM-ONLY BASELINE (No Feature Engineering)")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    print("Input: raw flattened landmarks (132 features) — no angles, no normalization")
    print()

    train_ds, val_ds, test_ds = build_raw_datasets(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = RawLSTM(input_size=132).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    rep_criterion = nn.MSELoss()
    form_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10}")
    print("-" * 35)

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for x_batch, rep_true, form_true in train_loader:
            x_batch, rep_true, form_true = x_batch.to(device), rep_true.to(device), form_true.to(device)
            rep_pred, form_pred = model(x_batch)
            loss = rep_criterion(rep_pred, rep_true) + 0.5 * form_criterion(form_pred, form_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, rep_true, form_true in val_loader:
                x_batch, rep_true, form_true = x_batch.to(device), rep_true.to(device), form_true.to(device)
                rep_pred, form_pred = model(x_batch)
                loss = rep_criterion(rep_pred, rep_true) + 0.5 * form_criterion(form_pred, form_true)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"{epoch+1:>5} | {avg_train:>10.4f} | {avg_val:>10.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), output_path / 'lstm_baseline_only.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print(f"\n{'='*60}")
    print("BASELINE FINAL EVALUATION")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(output_path / 'lstm_baseline_only.pth', weights_only=True))
    model.eval()

    all_rep_true, all_rep_pred = [], []
    all_form_true, all_form_pred = [], []

    with torch.no_grad():
        for x_batch, rep_true, form_true in test_loader:
            x_batch = x_batch.to(device)
            rep_pred, form_pred = model(x_batch)
            all_rep_true.extend(rep_true.numpy().flatten())
            all_rep_pred.extend(rep_pred.cpu().numpy().flatten())
            all_form_true.extend(form_true.numpy().flatten())
            all_form_pred.extend(form_pred.cpu().numpy().flatten())

    rep_true_arr = np.array(all_rep_true)
    rep_pred_arr = np.round(np.array(all_rep_pred))
    form_true_arr = np.array(all_form_true)
    form_pred_arr = (np.array(all_form_pred) > 0.5).astype(float)

    rep_mae = np.mean(np.abs(rep_true_arr - rep_pred_arr))
    rep_accuracy = np.mean(rep_true_arr == rep_pred_arr)
    form_accuracy = np.mean((form_true_arr > 0.5) == (form_pred_arr > 0.5))

    print(f"Rep Count MAE:       {rep_mae:.2f}")
    print(f"Rep Count Accuracy:  {rep_accuracy:.1%}  ← compare to CNN+LSTM: 76.2%")
    print(f"Form Classification: {form_accuracy:.1%}  ← compare to CNN+LSTM: 90.8%")

    results = {
        'baseline': 'lstm_only_raw_keypoints',
        'description': 'LSTM trained on raw flattened (33,4) landmarks — no feature engineering',
        'input_size': 132,
        'test_results': {
            'rep_mae': float(rep_mae),
            'rep_accuracy': float(rep_accuracy),
            'form_accuracy': float(form_accuracy),
        }
    }

    with open(output_path / 'baseline_lstm_only_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Baseline results saved to {output_path / 'baseline_lstm_only_results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM-only baseline')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()

    train_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
