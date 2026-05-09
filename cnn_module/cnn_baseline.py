"""
CNN-Only Baseline (Shaheer)
============================
Demonstrates what you get with ONLY spatial analysis (MediaPipe + heuristics),
NO temporal reasoning (no LSTM).

Approach:
  - Rep counting: Peak detection on the primary joint angle signal
  - Form classification: Simple min-angle threshold per exercise

This is Baseline 1 in the comparison table. It should perform worse than the
full CNN+LSTM pipeline because it lacks temporal context.

WHY THIS MATTERS (for viva):
  The CNN-only baseline proves that spatial keypoint extraction alone is
  insufficient for robust workout analysis. While peak detection can count
  reps in clean videos, it fails on:
    - Variable-speed reps
    - Pauses mid-rep
    - Noisy angle signals from occlusion
    - Multi-feature form assessment (e.g., back angle + knee angle together)

Usage:
  python -m cnn_module.cnn_baseline --data_dir data/processed
"""

import numpy as np
import json
import argparse
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import accuracy_score, confusion_matrix


# Primary angle index in the 84-feature vector for each exercise
PRIMARY_ANGLE_INDEX = {
    'squat': 0,         # left_knee angle
    'pushup': 4,        # left_elbow angle
    'hammer_curl': 4,   # left_elbow angle
}

# Form threshold: if min angle during a rep is below this → predicted "good form"
FORM_THRESHOLDS = {
    'squat': 100,       # Good squat = knee angle dips below 100°
    'pushup': 100,      # Good pushup = elbow angle dips below 100°
    'hammer_curl': 60,  # Good curl = elbow angle dips below 60°
}


def count_reps_peak_detection(angle_signal, min_distance=24, prominence=15):
    """
    Count reps using peak detection on the inverted angle signal.

    HOW IT WORKS (for viva):
      1. Smooth the angle signal with Savitzky-Golay filter to remove noise
      2. Invert the signal (negate it) so rep "bottoms" become peaks
      3. Use scipy.signal.find_peaks to find local maxima
      4. Each peak = one rep

    Args:
        angle_signal: 1D array of joint angle values over time
        min_distance: minimum frames between two reps (~0.8 sec at 30fps)
        prominence: minimum angle change to qualify as a rep (degrees)

    Returns:
        rep_count: number of reps detected
        peak_frames: frame indices where reps bottomed out
    """
    if len(angle_signal) < 15:
        return 0, []

    # Step 1: Smooth the signal
    window_len = min(15, len(angle_signal))
    if window_len % 2 == 0:
        window_len -= 1
    if window_len < 5:
        return 0, []

    smoothed = savgol_filter(angle_signal, window_len, 3)

    # Step 2: Find peaks on inverted signal (bottoms of reps)
    peaks, properties = find_peaks(
        -smoothed,
        distance=min_distance,
        prominence=prominence
    )

    return len(peaks), peaks.tolist()


def classify_form_by_threshold(angle_signal, exercise_type):
    """
    Classify form quality using a single min-angle threshold.

    This is deliberately simple — it's the BASELINE. It only checks
    whether the angle went low enough (depth check). It ignores:
      - Back angle (posture)
      - Velocity profile (smoothness)
      - Temporal consistency
      - Multi-joint coordination

    That's why the CNN+LSTM pipeline should outperform this.
    """
    threshold = FORM_THRESHOLDS.get(exercise_type, 100)
    min_angle = float(np.min(angle_signal))

    # Good form = angle dipped below threshold (sufficient depth/contraction)
    return 1 if min_angle < threshold else 0


def evaluate_cnn_baseline(data_dir):
    """
    Run the CNN-only baseline on all processed data.
    Reports per-exercise and overall metrics.
    """
    processed_path = Path(data_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")

    # Per-exercise tracking
    exercise_results = {}
    overall = {
        'true_reps': [], 'pred_reps': [],
        'true_form': [], 'pred_form': [],
    }

    for exercise_dir in sorted(processed_path.iterdir()):
        if not exercise_dir.is_dir() or exercise_dir.name.startswith('.'):
            continue

        exercise_type = exercise_dir.name
        ex_true_reps, ex_pred_reps = [], []
        ex_true_form, ex_pred_form = [], []

        label_files = sorted(exercise_dir.glob("*_labels.json"))

        for label_file in label_files:
            feat_file = label_file.parent / label_file.name.replace(
                '_labels.json', '_features.npy'
            )
            if not feat_file.exists():
                continue

            with open(label_file) as f:
                labels = json.load(f)

            features = np.load(feat_file)
            exercise = labels['exercise_type']

            # Get primary angle signal
            angle_idx = PRIMARY_ANGLE_INDEX.get(exercise, 0)
            angle_signal = features[:, angle_idx]

            # Rep counting via peak detection
            pred_reps, peak_frames = count_reps_peak_detection(angle_signal)
            true_reps = labels['rep_count']

            ex_true_reps.append(true_reps)
            ex_pred_reps.append(pred_reps)

            # Form classification via threshold
            if labels['form_labels']:
                for fl in labels['form_labels']:
                    form_pred = classify_form_by_threshold(angle_signal, exercise)
                    ex_true_form.append(fl)
                    ex_pred_form.append(form_pred)

        # Per-exercise metrics
        if ex_true_reps:
            true_arr = np.array(ex_true_reps)
            pred_arr = np.array(ex_pred_reps)
            mae = np.mean(np.abs(true_arr - pred_arr))
            acc = np.mean(true_arr == pred_arr)
            form_acc = accuracy_score(ex_true_form, ex_pred_form) if ex_true_form else 0

            exercise_results[exercise_type] = {
                'num_videos': len(ex_true_reps),
                'total_true_reps': int(true_arr.sum()),
                'total_pred_reps': int(pred_arr.sum()),
                'rep_mae': float(mae),
                'rep_accuracy': float(acc),
                'form_accuracy': float(form_acc),
            }

            overall['true_reps'].extend(ex_true_reps)
            overall['pred_reps'].extend(ex_pred_reps)
            overall['true_form'].extend(ex_true_form)
            overall['pred_form'].extend(ex_pred_form)

    # Overall metrics
    true_reps = np.array(overall['true_reps'])
    pred_reps = np.array(overall['pred_reps'])
    overall_mae = np.mean(np.abs(true_reps - pred_reps))
    overall_rep_acc = np.mean(true_reps == pred_reps)
    overall_form_acc = accuracy_score(
        overall['true_form'], overall['pred_form']
    ) if overall['true_form'] else 0

    # Confusion matrix for form classification
    if overall['true_form']:
        cm = confusion_matrix(overall['true_form'], overall['pred_form'])
    else:
        cm = None

    # ===== PRINT REPORT =====
    print("\n" + "=" * 65)
    print("CNN-ONLY BASELINE EVALUATION REPORT")
    print("Method: MediaPipe Keypoints + Peak Detection + Angle Thresholds")
    print("=" * 65)

    print(f"\n{'Exercise':<15} {'Videos':<8} {'True Reps':<12} {'Pred Reps':<12} "
          f"{'Rep MAE':<10} {'Rep Acc':<10} {'Form Acc':<10}")
    print("-" * 77)

    for exercise, stats in exercise_results.items():
        print(f"{exercise:<15} {stats['num_videos']:<8} "
              f"{stats['total_true_reps']:<12} {stats['total_pred_reps']:<12} "
              f"{stats['rep_mae']:<10.2f} {stats['rep_accuracy']:<10.1%} "
              f"{stats['form_accuracy']:<10.1%}")

    print("-" * 77)
    print(f"{'OVERALL':<15} {len(overall['true_reps']):<8} "
          f"{int(true_reps.sum()):<12} {int(pred_reps.sum()):<12} "
          f"{overall_mae:<10.2f} {overall_rep_acc:<10.1%} "
          f"{overall_form_acc:<10.1%}")

    if cm is not None:
        print(f"\nForm Classification Confusion Matrix:")
        print(f"                    Predicted Bad  Predicted Good")
        print(f"  Actual Bad   {cm[0][0]:>12}  {cm[0][1]:>14}")
        print(f"  Actual Good  {cm[1][0]:>12}  {cm[1][1]:>14}")

    print("\n" + "=" * 65)
    print("LIMITATIONS OF THIS BASELINE (use these in your presentation):")
    print("=" * 65)
    print("  1. Peak detection fails on variable-speed reps")
    print("  2. Single-angle threshold ignores multi-joint form assessment")
    print("  3. No temporal context — can't detect mid-rep pauses or jerky motion")
    print("  4. Static thresholds don't adapt to different body proportions")
    print("  5. Cannot learn from data — only as good as hand-tuned parameters")

    # Save results
    results = {
        'method': 'CNN-Only (Peak Detection + Angle Threshold)',
        'per_exercise': exercise_results,
        'overall': {
            'rep_mae': float(overall_mae),
            'rep_accuracy': float(overall_rep_acc),
            'form_accuracy': float(overall_form_acc),
            'total_videos': len(overall['true_reps']),
            'total_true_reps': int(true_reps.sum()),
            'total_pred_reps': int(pred_reps.sum()),
        },
    }

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'cnn_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to results/cnn_baseline_results.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN-Only Baseline Evaluation')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    args = parser.parse_args()

    evaluate_cnn_baseline(args.data_dir)
