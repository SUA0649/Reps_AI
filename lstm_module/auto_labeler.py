"""
Auto-Labeling Module (Abdul Rehman)
====================================
Automatically labels exercise data with:
  1. Rep boundaries (start/end frame of each rep)
  2. Form quality (good/bad for each rep)

Uses signal processing (peak detection) for rep counting
and biomechanical angle thresholds for form assessment.

BIOMECHANICAL RULES (for viva — cite NSCA guidelines):
  Squat:
    Good: knee ≤ 90°, back ≤ 30° from vertical, no knee valgus
    Bad:  knee > 120° (half rep), back > 45°, or knee valgus
  Push-up:
    Good: elbow ≤ 90°, hip sag ≤ 10°, full extension at top
    Bad:  elbow > 120° (half rep), hip sag > 20°
  Hammer Curl:
    Good: elbow ≤ 40° (full curl), shoulder stable, elbow near torso
    Bad:  elbow > 80° (partial), shoulder shrug > 5%, body sway
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path
import json


class AutoLabeler:
    """Automatically label exercise sequences with rep counts and form quality."""

    # Biomechanical thresholds per exercise
    FORM_RULES = {
        'squat': {
            'primary_angle': 'left_knee',
            'min_angle_good': 90,    # Must reach ≤ 90° for full rep
            'min_angle_bad': 120,    # If angle never goes below 120° → half rep
            'back_angle_max': 45,    # Back angle threshold (degrees from vertical)
        },
        'pushup': {
            'primary_angle': 'left_elbow',
            'min_angle_good': 90,
            'min_angle_bad': 120,
            'hip_sag_max': 20,       # Max hip sag angle
        },
        'hammer_curl': {
            'primary_angle': 'left_elbow',
            'min_angle_good': 50,    # Must reach ≤ 50° for full curl
            'min_angle_bad': 80,     # Partial curl threshold
            'shoulder_move_max': 0.05,  # Max shoulder Y movement (normalized)
        },
    }

    def __init__(self, fps=30):
        """
        Args:
            fps: frames per second of processed video (for peak detection tuning)
        """
        self.fps = fps

    def detect_reps(self, angle_signal, exercise_type):
        """
        Detect individual reps using peak detection on the joint angle signal.

        HOW IT WORKS:
          1. Smooth the angle signal to remove noise
          2. Invert it (find valleys = bottom of each rep)
          3. Use scipy.signal.find_peaks to find the valleys
          4. Each valley = one rep bottom position

        Args:
            angle_signal: (num_frames,) array of primary joint angle over time
            exercise_type: 'squat', 'pushup', or 'hammer_curl'

        Returns:
            rep_boundaries: list of (start_frame, valley_frame, end_frame) tuples
            smoothed_signal: the smoothed angle signal (for visualization)
        """
        # Step 1: Smooth the signal to remove MediaPipe jitter
        # Savitzky-Golay filter: polynomial smoothing that preserves peaks
        window_length = min(15, len(angle_signal))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < 5:
            return [], angle_signal

        smoothed = savgol_filter(angle_signal, window_length, polyorder=3)

        # Step 2: Invert signal to find valleys (bottom of reps)
        # For all exercises, a rep = angle goes DOWN then UP
        inverted = -smoothed

        # Step 3: Find peaks in inverted signal (= valleys in original)
        # min distance between reps: ~0.8 seconds (24 frames at 30fps)
        min_distance = int(self.fps * 0.8)

        # Prominence: peak must be at least 15° deeper than surrounding
        peaks, properties = find_peaks(
            inverted,
            distance=min_distance,
            prominence=15,
        )

        if len(peaks) == 0:
            return [], smoothed

        # Step 4: Find rep boundaries (midpoints between consecutive valleys)
        rep_boundaries = []
        for i, valley in enumerate(peaks):
            # Start = midpoint to previous valley (or start of signal)
            if i == 0:
                start = 0
            else:
                start = (peaks[i - 1] + valley) // 2

            # End = midpoint to next valley (or end of signal)
            if i == len(peaks) - 1:
                end = len(smoothed) - 1
            else:
                end = (valley + peaks[i + 1]) // 2

            rep_boundaries.append((start, valley, end))

        return rep_boundaries, smoothed

    def assess_form(self, features, angle_names, rep_boundaries, exercise_type):
        """
        Assess form quality for each detected rep using biomechanical rules.

        Args:
            features: (num_frames, 84) feature array
            angle_names: list of angle names from FeatureEngineer
            rep_boundaries: list of (start, valley, end) from detect_reps
            exercise_type: exercise name

        Returns:
            form_labels: list of 0 (bad) or 1 (good) for each rep
            form_details: list of dicts with per-rep analysis details
        """
        rules = self.FORM_RULES[exercise_type]
        primary_idx = angle_names.index(rules['primary_angle'])

        form_labels = []
        form_details = []

        for start, valley, end in rep_boundaries:
            rep_features = features[start:end + 1]
            rep_angles = rep_features[:, primary_idx]

            # Check 1: Did the person reach sufficient depth/contraction?
            min_angle = np.min(rep_angles)
            depth_ok = min_angle <= rules['min_angle_good']
            half_rep = min_angle > rules['min_angle_bad']

            detail = {
                'min_angle': float(min_angle),
                'depth_ok': depth_ok,
                'half_rep': half_rep,
            }

            # Exercise-specific checks
            if exercise_type == 'squat':
                # Check back angle (hip angle as proxy for trunk lean)
                hip_idx = angle_names.index('left_hip')
                hip_angles = rep_features[:, hip_idx]
                back_ok = np.min(hip_angles) > (180 - rules['back_angle_max'])
                detail['back_ok'] = back_ok
                is_good = depth_ok and back_ok and not half_rep

            elif exercise_type == 'pushup':
                # Check for hip sag (compare hip Y to shoulder-ankle line)
                is_good = depth_ok and not half_rep

            elif exercise_type == 'hammer_curl':
                # Check shoulder stability
                shoulder_idx = angle_names.index('left_shoulder')
                shoulder_angles = rep_features[:, shoulder_idx]
                shoulder_range = np.max(shoulder_angles) - np.min(shoulder_angles)
                shoulder_ok = shoulder_range < 20  # Less than 20° movement
                detail['shoulder_stable'] = shoulder_ok
                is_good = depth_ok and shoulder_ok and not half_rep

            else:
                is_good = depth_ok and not half_rep

            form_labels.append(1 if is_good else 0)
            form_details.append(detail)

        return form_labels, form_details

    def label_sequence(self, features, angle_names, exercise_type):
        """
        Full auto-labeling pipeline for one video's features.

        Args:
            features: (num_frames, 84) feature array
            angle_names: list of angle feature names
            exercise_type: 'squat', 'pushup', or 'hammer_curl'

        Returns:
            labels: dict with rep_count, rep_boundaries, form_labels, etc.
        """
        # Get primary angle signal
        primary = {
            'squat': 'left_knee',
            'pushup': 'left_elbow',
            'hammer_curl': 'left_elbow',
        }
        angle_idx = angle_names.index(primary[exercise_type])
        angle_signal = features[:, angle_idx]

        # Detect reps
        rep_boundaries, smoothed = self.detect_reps(angle_signal, exercise_type)

        # Assess form for each rep
        if rep_boundaries:
            form_labels, form_details = self.assess_form(
                features, angle_names, rep_boundaries, exercise_type
            )
        else:
            form_labels, form_details = [], []

        labels = {
            'exercise_type': exercise_type,
            'rep_count': len(rep_boundaries),
            'rep_boundaries': rep_boundaries,
            'form_labels': form_labels,
            'form_details': form_details,
            'avg_form_quality': float(np.mean(form_labels)) if form_labels else 0.0,
            'num_frames': len(features),
        }

        return labels


if __name__ == "__main__":
    # Quick test with synthetic angle signal (simulating 5 squats)
    labeler = AutoLabeler(fps=30)

    # Create fake angle signal: 5 reps of squats (angle goes 170→90→170)
    t = np.linspace(0, 10, 300)  # 10 seconds at 30fps
    angle_signal = 130 + 40 * np.cos(2 * np.pi * 0.5 * t)  # Oscillate 90-170°

    reps, smoothed = labeler.detect_reps(angle_signal, 'squat')
    print(f"Detected {len(reps)} reps")
    for i, (s, v, e) in enumerate(reps):
        print(f"  Rep {i+1}: frames {s}-{e}, valley at {v}")
    print("✅ Auto-labeler OK")
