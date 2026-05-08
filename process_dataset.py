"""
Main Dataset Processing Script (Shaheer)
=========================================
Orchestrates the full data pipeline:
  1. Scan Kaggle dataset folders
  2. Extract keypoints from each video using MediaPipe
  3. Compute features (angles, coordinates, velocities)
  4. Auto-label reps and form quality
  5. Save processed data for LSTM training

Usage:
  python process_dataset.py --data_dir data/kaggle_exercise_videos --output_dir data/processed

Expected Kaggle dataset structure:
  data/kaggle_exercise_videos/
    ├── squat/
    │   ├── video1.mp4
    │   └── video2.mp4
    ├── pushup/     (or push-up, push_up — we handle variations)
    │   └── ...
    └── hammer_curl/ (or hammer-curl, hammercurl)
        └── ...
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from cnn_module.keypoint_extractor import KeypointExtractor
from cnn_module.feature_engineer import FeatureEngineer
from lstm_module.auto_labeler import AutoLabeler


# Map various folder name formats to our standard exercise names
EXERCISE_NAME_MAP = {
    'squat': 'squat',
    'squats': 'squat',
    'pushup': 'pushup',
    'push-up': 'pushup',
    'push_up': 'pushup',
    'pushups': 'pushup',
    'push-ups': 'pushup',
    'push_ups': 'pushup',
    'hammer_curl': 'hammer_curl',
    'hammer-curl': 'hammer_curl',
    'hammercurl': 'hammer_curl',
    'hammer_curls': 'hammer_curl',
    'hammer curl': 'hammer_curl',
}


def process_dataset(data_dir, output_dir, target_fps=30):
    """
    Process the entire Kaggle exercise video dataset.

    Args:
        data_dir: path to kaggle_exercise_videos/
        output_dir: path to data/processed/
        target_fps: standardize all videos to this FPS
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    # Initialize modules
    extractor = KeypointExtractor(model_complexity=1)
    feature_eng = FeatureEngineer()
    labeler = AutoLabeler(fps=target_fps)

    # Scan for exercise folders
    exercise_folders = [f for f in data_path.iterdir() if f.is_dir()]
    print(f"Found {len(exercise_folders)} exercise folders: {[f.name for f in exercise_folders]}")

    total_processed = 0
    total_failed = 0
    summary = {}

    for folder in exercise_folders:
        # Map folder name to standard exercise name
        folder_name = folder.name.lower().strip()
        exercise_type = EXERCISE_NAME_MAP.get(folder_name)

        if exercise_type is None:
            print(f"⚠️ Unknown exercise folder '{folder.name}', skipping")
            print(f"   Known exercises: {list(set(EXERCISE_NAME_MAP.values()))}")
            continue

        # Create output directory for this exercise
        exercise_output = output_path / exercise_type
        exercise_output.mkdir(parents=True, exist_ok=True)

        # Find video files
        video_files = sorted(
            list(folder.glob("*.mp4")) +
            list(folder.glob("*.avi")) +
            list(folder.glob("*.mov")) +
            list(folder.glob("*.MP4"))
        )
        print(f"\n{'='*60}")
        print(f"Processing {exercise_type}: {len(video_files)} videos")
        print(f"{'='*60}")

        exercise_stats = {'processed': 0, 'failed': 0, 'total_reps': 0}

        for video_file in tqdm(video_files, desc=exercise_type):
            video_name = video_file.stem  # filename without extension

            try:
                # Step 1: Extract keypoints
                keypoints, metadata = extractor.extract_from_video(
                    video_file, target_fps=target_fps
                )

                if keypoints.shape[0] < 30:  # Less than 1 second of data
                    print(f"  ⚠️ {video_name}: too short ({keypoints.shape[0]} frames), skipping")
                    exercise_stats['failed'] += 1
                    continue

                # Step 2: Compute features
                features, angle_names = feature_eng.extract_features_sequence(keypoints)

                # Step 3: Auto-label
                labels = labeler.label_sequence(features, angle_names, exercise_type)

                # Step 4: Save processed data
                np.save(exercise_output / f"{video_name}_keypoints.npy", keypoints)
                np.save(exercise_output / f"{video_name}_features.npy", features)

                with open(exercise_output / f"{video_name}_labels.json", 'w') as f:
                    # Convert numpy types for JSON serialization
                    serializable_labels = {
                        'exercise_type': labels['exercise_type'],
                        'rep_count': int(labels['rep_count']),
                        'rep_boundaries': [(int(s), int(v), int(e))
                                          for s, v, e in labels['rep_boundaries']],
                        'form_labels': [int(fl) for fl in labels['form_labels']],
                        'form_details': labels['form_details'],
                        'avg_form_quality': float(labels['avg_form_quality']),
                        'num_frames': int(labels['num_frames']),
                    }
                    json.dump(serializable_labels, f, indent=2)

                exercise_stats['processed'] += 1
                exercise_stats['total_reps'] += labels['rep_count']
                total_processed += 1

            except Exception as e:
                print(f"  ❌ {video_name}: {str(e)}")
                exercise_stats['failed'] += 1
                total_failed += 1

        summary[exercise_type] = exercise_stats
        print(f"  ✅ {exercise_stats['processed']} processed, "
              f"{exercise_stats['failed']} failed, "
              f"{exercise_stats['total_reps']} total reps detected")

    # Save angle names for later use
    with open(output_path / 'angle_names.json', 'w') as f:
        json.dump(angle_names, f)

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {total_processed}")
    print(f"Total failed:    {total_failed}")
    for exercise, stats in summary.items():
        print(f"  {exercise}: {stats['processed']} videos, {stats['total_reps']} reps")

    # Save summary
    with open(output_path / 'processing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    extractor.close()
    print(f"\n✅ Processing complete! Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process exercise video dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to Kaggle exercise video dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS for standardization')
    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_dir, target_fps=args.fps)
