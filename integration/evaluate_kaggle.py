"""
Kaggle Exercise Pose Dataset Validation (Both)
===============================================
Evaluates the full CNN+LSTM pipeline against the Kaggle Exercise Pose
validation set (.avi videos organized in exercise subfolders).

Expected folder structure:
  val/
  ├── squat/
  │   ├── video1.avi
  │   └── video2.avi
  ├── push-up/
  │   ├── video1.avi
  │   └── ...
  └── other_exercise/
      └── ...

Only evaluates exercises that exist in our training set
(squat, pushup, pullup). Other folders are reported as "not in training set".

Usage:
  python -m integration.evaluate_kaggle
"""

import numpy as np
import pandas as pd
import torch
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from cnn_module.keypoint_extractor import KeypointExtractor
from cnn_module.feature_engineer import FeatureEngineer
from lstm_module.auto_labeler import AutoLabeler
from lstm_module.model import ExerciseLSTM
from lstm_module.dataset import create_sliding_windows

# ==========================================
# USER: UPDATE THESE PATHS!
KAGGLE_VAL_CSV = "/Users/shaheeruddinahmed/Downloads/archive (1)/val.csv"
KAGGLE_VAL_VIDEOS = "/Users/shaheeruddinahmed/Downloads/archive (1)/val"
LSTM_MODEL_PATH = "models/lstm_best.pth"
# ==========================================

# Exercises we have trained on (LSTM-supported)
TRAINED_EXERCISES = {'squat', 'pushup', 'pullup'}

# Map common folder names to our internal exercise names
FOLDER_NAME_MAP = {
    'squat': 'squat', 'squats': 'squat',
    'push-up': 'pushup', 'push up': 'pushup', 'pushup': 'pushup',
    'pushups': 'pushup', 'push_up': 'pushup', 'push-ups': 'pushup',
    'pull-up': 'pullup', 'pull up': 'pullup', 'pullup': 'pullup',
    'pullups': 'pullup', 'pull_up': 'pullup', 'pull-ups': 'pullup',
    'plank': 'plank', 'planks': 'plank',
}


def evaluate_kaggle(videos_dir, csv_path=None, model_path=LSTM_MODEL_PATH):
    """
    Evaluate the full CNN+LSTM pipeline on the Kaggle validation set.
    
    Args:
        videos_dir: Path to validation folder containing exercise subfolders
        csv_path: Optional CSV with ground truth (filename, label, reps)
        model_path: Path to trained LSTM weights
    """
    print(f"Scanning validation folder: {videos_dir}")
    if not os.path.exists(videos_dir):
        print("❌ Error: Update KAGGLE_VAL_VIDEOS in the script with the actual path!")
        return

    val_path = Path(videos_dir)

    # Discover exercise subfolders
    all_folders = [f for f in val_path.iterdir() if f.is_dir()]
    if not all_folders:
        print("❌ No exercise subfolders found in validation directory!")
        return

    print(f"Found {len(all_folders)} exercise folders: {[f.name for f in all_folders]}")

    # Classify folders: trained vs. not trained
    trained_folders = {}
    skipped_folders = []

    for folder in all_folders:
        folder_key = folder.name.lower().strip()
        exercise = FOLDER_NAME_MAP.get(folder_key)

        if exercise and exercise in TRAINED_EXERCISES:
            trained_folders[folder] = exercise
        else:
            skipped_folders.append((folder.name, exercise or 'unknown'))

    # Report skipped exercises
    if skipped_folders:
        print(f"\n⚠️ The following folders are NOT in our training set (skipping):")
        for name, mapped in skipped_folders:
            if mapped == 'plank':
                print(f"   📁 {name} → '{mapped}' (handled by rule-based system, not LSTM)")
            else:
                print(f"   📁 {name} → NOT FOUND in training set")

    # Check if any trained exercises are missing from validation
    found_exercises = set(trained_folders.values())
    missing = TRAINED_EXERCISES - found_exercises
    if missing:
        print(f"\n⚠️ Trained exercises NOT found in validation data: {missing}")

    if not trained_folders:
        print("\n❌ No matching exercises found between training set and validation data!")
        return

    print(f"\n✅ Will evaluate: {list(set(trained_folders.values()))}")

    # Load optional CSV for ground truth reps
    csv_reps = {}
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nCSV loaded: {len(df)} rows, columns: {list(df.columns)}")
        # Try to build a filename → reps lookup
        for col_name_candidates in [
            ('filename', 'reps'), ('video', 'rep_count'), 
            ('file', 'repetitions'), ('name', 'count')
        ]:
            file_col = next((c for c in df.columns if c.lower().strip() == col_name_candidates[0]), None)
            reps_col = next((c for c in df.columns if c.lower().strip() == col_name_candidates[1]), None)
            if file_col and reps_col:
                for _, row in df.iterrows():
                    csv_reps[str(row[file_col])] = int(row[reps_col])
                print(f"   Loaded {len(csv_reps)} rep count annotations")
                break

    # Initialize pipeline
    extractor = KeypointExtractor()
    feature_eng = FeatureEngineer()
    labeler = AutoLabeler(fps=30)

    # Load LSTM model
    lstm_available = False
    if os.path.exists(model_path):
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = ExerciseLSTM(input_size=84).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        lstm_available = True
        print(f"   LSTM model loaded from {model_path} (device: {device})")
    else:
        print(f"   ⚠️ LSTM model not found at {model_path}, running CNN-only evaluation")

    # ===== PER-EXERCISE EVALUATION =====
    overall_results = {}

    for folder, exercise in trained_folders.items():
        videos = list(folder.glob("*.avi")) + list(folder.glob("*.mp4")) + list(folder.glob("*.mov"))
        if not videos:
            print(f"\n⚠️ {folder.name}: No .avi/.mp4/.mov videos found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating {exercise.upper()} ({len(videos)} videos from '{folder.name}/')")
        print(f"{'='*50}")

        ex_processed = 0
        ex_failed = 0
        ex_true_reps = []
        ex_pred_reps = []
        ex_form_scores = []
        ex_total_pred_reps = 0

        for video_path in tqdm(videos, desc=exercise):
            try:
                # Step 1: Extract keypoints
                keypoints, metadata = extractor.extract_from_video(str(video_path), target_fps=30)

                if keypoints.shape[0] < 30:
                    ex_failed += 1
                    continue

                # Step 2: Compute features
                features, angle_names = feature_eng.extract_features_sequence(keypoints)

                # Step 3: Auto-label (CNN-based rep counting)
                labels = labeler.label_sequence(features, angle_names, exercise)
                pred_reps = labels['rep_count']
                ex_total_pred_reps += pred_reps

                # Check if CSV has ground truth reps for this video
                video_key = video_path.name
                if video_key in csv_reps:
                    ex_true_reps.append(csv_reps[video_key])
                    ex_pred_reps.append(pred_reps)

                # Step 4: LSTM form classification
                if lstm_available:
                    windows, _ = create_sliding_windows(features,
                        labels, window_size=60, stride=30)

                    if windows:
                        window_forms = []
                        for window in windows:
                            x = torch.FloatTensor(window).unsqueeze(0).to(device)
                            with torch.no_grad():
                                _, form_pred = model(x)
                            window_forms.append(form_pred.item())
                        ex_form_scores.append(np.mean(window_forms))

                ex_processed += 1

            except Exception as e:
                ex_failed += 1
                if ex_failed <= 3:
                    print(f"  ❌ {video_path.name}: {e}")

        # Per-exercise summary
        print(f"\n  Processed: {ex_processed}/{len(videos)}, Failed: {ex_failed}")
        print(f"  Total Predicted Reps: {ex_total_pred_reps}")

        result = {
            'videos': len(videos),
            'processed': ex_processed,
            'failed': ex_failed,
            'total_pred_reps': ex_total_pred_reps,
        }

        if ex_true_reps:
            true_arr = np.array(ex_true_reps)
            pred_arr = np.array(ex_pred_reps)
            mae = np.mean(np.abs(true_arr - pred_arr))
            acc = np.mean(true_arr == pred_arr)
            print(f"  Rep Count MAE: {mae:.2f}")
            print(f"  Rep Exact Match: {acc:.1%}")
            result['rep_mae'] = float(mae)
            result['rep_accuracy'] = float(acc)

        if ex_form_scores:
            avg_form = np.mean(ex_form_scores)
            good_pct = np.mean([1 if s > 0.5 else 0 for s in ex_form_scores])
            print(f"  Avg Form Score: {avg_form:.3f}")
            print(f"  Good Form Rate: {good_pct:.1%}")
            result['avg_form_score'] = float(avg_form)
            result['good_form_rate'] = float(good_pct)

        overall_results[exercise] = result

    # ===== FINAL SUMMARY =====
    print(f"\n\n{'='*60}")
    print("KAGGLE VALIDATION — FINAL SUMMARY")
    print(f"{'='*60}")

    total_videos = sum(r['processed'] for r in overall_results.values())
    total_reps = sum(r['total_pred_reps'] for r in overall_results.values())

    print(f"\n{'Exercise':<12} {'Videos':<8} {'Reps':<8} {'Form Score':<12} {'Good Form':<10}")
    print("-" * 50)
    for ex, r in overall_results.items():
        form_str = f"{r.get('avg_form_score', 0):.3f}" if 'avg_form_score' in r else "N/A"
        good_str = f"{r.get('good_form_rate', 0):.1%}" if 'good_form_rate' in r else "N/A"
        print(f"{ex:<12} {r['processed']:<8} {r['total_pred_reps']:<8} {form_str:<12} {good_str:<10}")

    print("-" * 50)
    print(f"{'TOTAL':<12} {total_videos:<8} {total_reps:<8}")

    if skipped_folders:
        print(f"\nSkipped folders (not in training set):")
        for name, mapped in skipped_folders:
            print(f"  📁 {name}")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'kaggle_validation_results.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    print(f"\n✅ Results saved to results/kaggle_validation_results.json")


if __name__ == "__main__":
    evaluate_kaggle(KAGGLE_VAL_VIDEOS, KAGGLE_VAL_CSV)
