import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_labels(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Directory '{data_dir}' does not exist.")
        return
        
    stats = defaultdict(lambda: {'good_form': 0, 'bad_form': 0, 'total_reps': 0, 'videos': 0})
    
    total_good = 0
    total_bad = 0
    
    # Iterate through each exercise folder in the processed data directory
    for exercise_folder in data_path.iterdir():
        if not exercise_folder.is_dir():
            continue
            
        exercise = exercise_folder.name
        json_files = list(exercise_folder.glob("*_labels.json"))
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Failed to read {json_file}: {e}")
                    continue
                    
                stats[exercise]['videos'] += 1
                
                # Check form labels (1 = Good, 0 = Bad)
                form_labels = data.get('form_labels', [])
                for label in form_labels:
                    if label == 1:
                        stats[exercise]['good_form'] += 1
                        total_good += 1
                    else:
                        stats[exercise]['bad_form'] += 1
                        total_bad += 1
                        
                    stats[exercise]['total_reps'] += 1

    print("="*50)
    print("🏋️  DATASET FORM QUALITY SUMMARY 🏋️")
    print("="*50)
    
    for exercise, counts in stats.items():
        print(f"\nExercise: {exercise.upper()} ({counts['videos']} videos)")
        print(f"  Total Reps: {counts['total_reps']}")
        if counts['total_reps'] > 0:
            good_pct = (counts['good_form'] / counts['total_reps']) * 100
            bad_pct = (counts['bad_form'] / counts['total_reps']) * 100
            print(f"  Good Form:  {counts['good_form']} ({good_pct:.1f}%)")
            print(f"  Bad Form:   {counts['bad_form']} ({bad_pct:.1f}%)")
        else:
            print("  No reps found.")
            
    print("\n" + "="*50)
    print(f"TOTAL DATASET REPS: {total_good + total_bad}")
    if (total_good + total_bad) > 0:
        print(f"TOTAL GOOD FORM:    {total_good} ({(total_good/(total_good+total_bad))*100:.1f}%)")
        print(f"TOTAL BAD FORM:     {total_bad} ({(total_bad/(total_good+total_bad))*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze form quality distribution in processed dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to the processed data directory')
    args = parser.parse_args()
    
    # Resolve relative to the repository root if running from inside integration/
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_dir
    
    analyze_labels(data_path)
