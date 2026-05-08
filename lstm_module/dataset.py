"""
PyTorch Dataset for Exercise Sequences (Abdul Rehman)
=====================================================
Creates sliding window samples from processed video features.

Each sample = 60 frames (2 seconds at 30fps) of feature vectors,
labeled with rep count and form quality.

DATA AUGMENTATION (for viva):
  - Speed jitter: randomly stretch/compress sequences (0.8x-1.2x)
  - Gaussian noise: add small noise to features (simulates sensor jitter)
  These increase effective dataset size without recording more videos.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class ExerciseDataset(Dataset):
    """Sliding window dataset for LSTM training."""

    def __init__(self, samples, labels, augment=False):
        """
        Args:
            samples: list of numpy arrays, each (window_size, num_features)
            labels: list of dicts with 'rep_count' and 'form_quality'
            augment: if True, apply data augmentation
        """
        self.samples = samples
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()  # (window_size, num_features)

        if self.augment:
            x = self._augment(x)

        rep_count = self.labels[idx]['rep_count']
        form_quality = self.labels[idx]['form_quality']

        return (
            torch.FloatTensor(x),
            torch.FloatTensor([rep_count]),
            torch.FloatTensor([form_quality]),
        )

    def _augment(self, x):
        """Apply data augmentation to a sequence."""
        # 1. Gaussian noise (σ = 0.01)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

        # 2. Speed jitter: randomly resample to 80%-120% speed
        if np.random.random() < 0.5:
            speed_factor = np.random.uniform(0.8, 1.2)
            orig_len = len(x)
            new_len = int(orig_len * speed_factor)
            if new_len > 2:
                indices = np.linspace(0, orig_len - 1, new_len).astype(int)
                x_resampled = x[indices]
                # Pad or crop back to original length
                if len(x_resampled) >= orig_len:
                    x = x_resampled[:orig_len]
                else:
                    pad = np.zeros((orig_len - len(x_resampled), x.shape[1]))
                    x = np.vstack([x_resampled, pad])

        return x


def create_sliding_windows(features, labels_dict, window_size=60, stride=15):
    """
    Create sliding window samples from a full video's features.

    Args:
        features: (num_frames, 84) feature array for one video
        labels_dict: output from AutoLabeler.label_sequence()
        window_size: number of frames per sample (default 60 = 2sec)
        stride: step between windows (default 15 = 0.5sec)

    Returns:
        windows: list of (window_size, 84) arrays
        window_labels: list of dicts with rep_count and form_quality
    """
    num_frames = len(features)
    rep_boundaries = labels_dict.get('rep_boundaries', [])
    form_labels = labels_dict.get('form_labels', [])

    windows = []
    window_labels = []

    for start in range(0, num_frames - window_size + 1, stride):
        end = start + window_size
        window = features[start:end]

        # Count complete reps in this window
        reps_in_window = 0
        form_scores = []

        for i, (r_start, r_valley, r_end) in enumerate(rep_boundaries):
            # A rep is "in" this window if its valley falls within the window
            if start <= r_valley < end:
                reps_in_window += 1
                if i < len(form_labels):
                    form_scores.append(form_labels[i])

        # Average form quality (1.0 = all good, 0.0 = all bad)
        avg_form = float(np.mean(form_scores)) if form_scores else 0.5

        windows.append(window)
        window_labels.append({
            'rep_count': reps_in_window,
            'form_quality': avg_form,
        })

    return windows, window_labels


def build_datasets(processed_dir, window_size=60, stride=15,
                   test_size=0.15, val_size=0.15, random_state=42):
    """
    Build train/val/test datasets from all processed exercise data.

    Args:
        processed_dir: path to data/processed/ containing exercise subdirs
        window_size: sliding window size
        stride: sliding window stride
        test_size: fraction for test set
        val_size: fraction for validation set

    Returns:
        train_dataset, val_dataset, test_dataset: ExerciseDataset objects
    """
    processed_dir = Path(processed_dir)
    all_windows = []
    all_labels = []

    for exercise_dir in processed_dir.iterdir():
        if not exercise_dir.is_dir():
            continue

        exercise_type = exercise_dir.name
        print(f"Loading {exercise_type}...")

        # Find all feature files
        for feat_file in sorted(exercise_dir.glob("*_features.npy")):
            label_file = feat_file.parent / feat_file.name.replace(
                '_features.npy', '_labels.json'
            )

            if not label_file.exists():
                print(f"  ⚠️ Missing labels for {feat_file.name}, skipping")
                continue

            features = np.load(feat_file)
            with open(label_file) as f:
                labels_dict = json.load(f)

            windows, w_labels = create_sliding_windows(
                features, labels_dict, window_size, stride
            )

            all_windows.extend(windows)
            all_labels.extend(w_labels)

    print(f"Total samples: {len(all_windows)}")

    if len(all_windows) == 0:
        raise ValueError("No data found! Check processed_dir path.")

    # Split into train / (val+test)
    indices = list(range(len(all_windows)))
    train_idx, test_val_idx = train_test_split(
        indices, test_size=(test_size + val_size), random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        test_val_idx, test_size=test_size / (test_size + val_size),
        random_state=random_state
    )

    def subset(idxs, augment=False):
        return ExerciseDataset(
            [all_windows[i] for i in idxs],
            [all_labels[i] for i in idxs],
            augment=augment,
        )

    train_ds = subset(train_idx, augment=True)
    val_ds = subset(val_idx, augment=False)
    test_ds = subset(test_idx, augment=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_ds, val_ds, test_ds


import json  # needed for label loading

if __name__ == "__main__":
    # Test with synthetic data
    fake_features = np.random.randn(300, 84)
    fake_labels = {
        'rep_count': 5,
        'rep_boundaries': [(20, 30, 40), (60, 70, 80), (100, 110, 120),
                           (140, 150, 160), (180, 190, 200)],
        'form_labels': [1, 1, 0, 1, 1],
    }

    windows, w_labels = create_sliding_windows(fake_features, fake_labels)
    print(f"Created {len(windows)} windows from 300 frames")
    print(f"Sample label: {w_labels[0]}")

    ds = ExerciseDataset(windows, w_labels, augment=True)
    x, rep, form = ds[0]
    print(f"Sample shape: {x.shape}, rep: {rep.item():.1f}, form: {form.item():.2f}")
    print("✅ Dataset OK")
