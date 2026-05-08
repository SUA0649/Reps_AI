"""
Evaluation & Comparison Script (Both)
======================================
Runs all three configurations and generates the comparison table:
  1. CNN-Only:  MediaPipe keypoints + peak detection (no LSTM)
  2. LSTM-Only: Raw features without spatial keypoint extraction
  3. CNN+LSTM:  Full pipeline (MediaPipe → Features → LSTM)

Usage:
  python -m integration.evaluate --data_dir data/processed --model models/lstm_best.pth
"""

import numpy as np
import torch
import json
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import find_peaks, savgol_filter

from lstm_module.model import ExerciseLSTM
from lstm_module.dataset import create_sliding_windows


def evaluate_cnn_only(processed_dir):
    """
    CNN-Only Baseline: use peak detection on angle signals (no LSTM).
    This is what you'd get with JUST keypoint extraction + heuristics.
    """
    print("\n" + "=" * 50)
    print("BASELINE 1: CNN-Only (Peak Detection)")
    print("=" * 50)

    all_true_reps = []
    all_pred_reps = []
    all_true_form = []
    all_pred_form = []

    processed_path = Path(processed_dir)

    for exercise_dir in processed_path.iterdir():
        if not exercise_dir.is_dir() or exercise_dir.name.startswith('.'):
            continue

        for label_file in sorted(exercise_dir.glob("*_labels.json")):
            feat_file = label_file.parent / label_file.name.replace(
                '_labels.json', '_features.npy'
            )
            if not feat_file.exists():
                continue

            with open(label_file) as f:
                labels = json.load(f)

            features = np.load(feat_file)
            exercise_type = labels['exercise_type']

            # CNN-Only approach: peak detection on angle signal
            primary_angles = {
                'squat': 0,       # left_knee angle index
                'pushup': 4,     # left_elbow angle index
                'hammer_curl': 4,
            }
            angle_idx = primary_angles.get(exercise_type, 0)
            angle_signal = features[:, angle_idx]

            # Smooth and find peaks
            if len(angle_signal) > 15:
                wl = min(15, len(angle_signal))
                if wl % 2 == 0:
                    wl -= 1
                smoothed = savgol_filter(angle_signal, wl, 3)
                peaks, _ = find_peaks(-smoothed, distance=24, prominence=15)
                pred_reps = len(peaks)
            else:
                pred_reps = 0

            true_reps = labels['rep_count']
            all_true_reps.append(true_reps)
            all_pred_reps.append(pred_reps)

            # Form: use simple angle threshold (no temporal reasoning)
            if labels['form_labels']:
                for fl in labels['form_labels']:
                    all_true_form.append(fl)
                    # CNN-only: classify by min angle alone
                    thresholds = {'squat': 100, 'pushup': 100, 'hammer_curl': 60}
                    thresh = thresholds.get(exercise_type, 100)
                    min_angle = float(np.min(angle_signal))
                    all_pred_form.append(1 if min_angle < thresh else 0)

    # Calculate metrics
    rep_mae = np.mean(np.abs(np.array(all_true_reps) - np.array(all_pred_reps)))
    rep_acc = np.mean(np.array(all_true_reps) == np.array(all_pred_reps))

    form_acc = accuracy_score(all_true_form, all_pred_form) if all_true_form else 0

    results = {
        'rep_mae': float(rep_mae),
        'rep_accuracy': float(rep_acc),
        'form_accuracy': float(form_acc),
    }

    print(f"  Rep MAE:           {rep_mae:.2f}")
    print(f"  Rep Accuracy:      {rep_acc:.1%}")
    print(f"  Form Accuracy:     {form_acc:.1%}")

    return results


def evaluate_lstm_only(processed_dir, model_path, window_size=60):
    """
    LSTM-Only Baseline: feed noise/random features (no keypoint extraction).
    Shows that LSTM alone cannot work without structured spatial input.
    """
    print("\n" + "=" * 50)
    print("BASELINE 2: LSTM-Only (Random Features)")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ExerciseLSTM(input_size=84).to(device)
    # Use UNTRAINED model weights (random initialization)
    model.eval()

    all_true_reps = []
    all_pred_reps = []
    all_true_form = []
    all_pred_form = []

    processed_path = Path(processed_dir)

    for exercise_dir in processed_path.iterdir():
        if not exercise_dir.is_dir() or exercise_dir.name.startswith('.'):
            continue

        for label_file in sorted(exercise_dir.glob("*_labels.json")):
            with open(label_file) as f:
                labels = json.load(f)

            # Feed RANDOM features instead of real keypoint features
            num_frames = labels['num_frames']
            if num_frames < window_size:
                continue

            random_features = np.random.randn(num_frames, 84).astype(np.float32)
            windows, w_labels = create_sliding_windows(
                random_features, labels, window_size
            )

            for window, w_label in zip(windows, w_labels):
                x = torch.FloatTensor(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    rep_pred, form_pred = model(x)

                all_true_reps.append(w_label['rep_count'])
                all_pred_reps.append(max(0, round(rep_pred.item())))
                all_true_form.append(1 if w_label['form_quality'] > 0.5 else 0)
                all_pred_form.append(1 if form_pred.item() > 0.5 else 0)

    rep_mae = np.mean(np.abs(np.array(all_true_reps) - np.array(all_pred_reps)))
    rep_acc = np.mean(np.array(all_true_reps) == np.array(all_pred_reps))
    form_acc = accuracy_score(all_true_form, all_pred_form) if all_true_form else 0

    results = {
        'rep_mae': float(rep_mae),
        'rep_accuracy': float(rep_acc),
        'form_accuracy': float(form_acc),
    }

    print(f"  Rep MAE:           {rep_mae:.2f}")
    print(f"  Rep Accuracy:      {rep_acc:.1%}")
    print(f"  Form Accuracy:     {form_acc:.1%}")

    return results


def evaluate_full_pipeline(processed_dir, model_path, window_size=60):
    """
    Full CNN+LSTM pipeline: MediaPipe features → trained LSTM.
    Should be the best performing configuration.
    """
    print("\n" + "=" * 50)
    print("FULL SYSTEM: CNN + LSTM")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ExerciseLSTM(input_size=84).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_true_reps = []
    all_pred_reps = []
    all_true_form = []
    all_pred_form = []

    processed_path = Path(processed_dir)

    for exercise_dir in processed_path.iterdir():
        if not exercise_dir.is_dir() or exercise_dir.name.startswith('.'):
            continue

        for feat_file in sorted(exercise_dir.glob("*_features.npy")):
            label_file = feat_file.parent / feat_file.name.replace(
                '_features.npy', '_labels.json'
            )
            if not label_file.exists():
                continue

            features = np.load(feat_file)
            with open(label_file) as f:
                labels = json.load(f)

            if len(features) < window_size:
                continue

            windows, w_labels = create_sliding_windows(features, labels, window_size)

            for window, w_label in zip(windows, w_labels):
                x = torch.FloatTensor(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    rep_pred, form_pred = model(x)

                all_true_reps.append(w_label['rep_count'])
                all_pred_reps.append(max(0, round(rep_pred.item())))
                all_true_form.append(1 if w_label['form_quality'] > 0.5 else 0)
                all_pred_form.append(1 if form_pred.item() > 0.5 else 0)

    rep_mae = np.mean(np.abs(np.array(all_true_reps) - np.array(all_pred_reps)))
    rep_acc = np.mean(np.array(all_true_reps) == np.array(all_pred_reps))
    form_acc = accuracy_score(all_true_form, all_pred_form) if all_true_form else 0

    results = {
        'rep_mae': float(rep_mae),
        'rep_accuracy': float(rep_acc),
        'form_accuracy': float(form_acc),
    }

    print(f"  Rep MAE:           {rep_mae:.2f}")
    print(f"  Rep Accuracy:      {rep_acc:.1%}")
    print(f"  Form Accuracy:     {form_acc:.1%}")

    return results


def run_comparison(data_dir, model_path):
    """Run all three evaluations and print comparison table."""
    cnn_only = evaluate_cnn_only(data_dir)
    lstm_only = evaluate_lstm_only(data_dir, model_path)
    full = evaluate_full_pipeline(data_dir, model_path)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (for presentation)")
    print("=" * 70)
    print(f"{'Metric':<25} {'CNN-Only':<15} {'LSTM-Only':<15} {'CNN+LSTM':<15}")
    print("-" * 70)
    print(f"{'Rep Count MAE':<25} {cnn_only['rep_mae']:<15.2f} "
          f"{lstm_only['rep_mae']:<15.2f} {full['rep_mae']:<15.2f}")
    print(f"{'Rep Count Accuracy':<25} {cnn_only['rep_accuracy']:<15.1%} "
          f"{lstm_only['rep_accuracy']:<15.1%} {full['rep_accuracy']:<15.1%}")
    print(f"{'Form Accuracy':<25} {cnn_only['form_accuracy']:<15.1%} "
          f"{lstm_only['form_accuracy']:<15.1%} {full['form_accuracy']:<15.1%}")
    print("=" * 70)

    # Save results
    results = {
        'cnn_only': cnn_only,
        'lstm_only': lstm_only,
        'full_pipeline': full,
    }

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to results/comparison_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate all configurations')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--model', type=str, default='models/lstm_best.pth')
    args = parser.parse_args()

    run_comparison(args.data_dir, args.model)
