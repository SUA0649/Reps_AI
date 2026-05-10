"""
Feature Engineering Module (Shaheer)
=====================================
Transforms raw MediaPipe keypoints into features for the LSTM.

Features per frame:
  1. Joint angles (8 values)
  2. Normalized coordinates (52 values)
  3. Velocity features (24 values)
  Total: 84 features per frame
"""

import numpy as np


class FeatureEngineer:
    """Compute exercise-relevant features from raw MediaPipe landmarks."""

    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11,  'right_shoulder': 12,
        'left_elbow': 13,     'right_elbow': 14,
        'left_wrist': 15,     'right_wrist': 16,
        'left_hip': 23,       'right_hip': 24,
        'left_knee': 25,      'right_knee': 26,
        'left_ankle': 27,     'right_ankle': 28,
    }

    # Joint angle definitions: (point_a, vertex, point_c)
    ANGLE_DEFINITIONS = {
        'left_knee':     ('left_hip',      'left_knee',     'left_ankle'),
        'right_knee':    ('right_hip',     'right_knee',    'right_ankle'),
        'left_hip':      ('left_shoulder', 'left_hip',      'left_knee'),
        'right_hip':     ('right_shoulder','right_hip',     'right_knee'),
        'left_elbow':    ('left_shoulder', 'left_elbow',    'left_wrist'),
        'right_elbow':   ('right_shoulder','right_elbow',   'right_wrist'),
        'left_shoulder': ('left_hip',      'left_shoulder', 'left_elbow'),
        'right_shoulder':('right_hip',     'right_shoulder','right_elbow'),
    }

    VELOCITY_JOINTS = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    ]

    # 26 key body landmarks (excluding detailed face landmarks)
    KEY_LANDMARK_INDICES = [
        0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
        17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 7, 8, 9
    ]

    def __init__(self):
        self.num_angle_features = len(self.ANGLE_DEFINITIONS)       # 8
        self.num_coord_features = len(self.KEY_LANDMARK_INDICES) * 2 # 52
        self.num_velocity_features = len(self.VELOCITY_JOINTS) * 2   # 24
        self.total_features = (self.num_angle_features +
                               self.num_coord_features +
                               self.num_velocity_features)           # 84

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate angle at vertex B given points A, B, C.
        Uses dot product formula: angle = arccos(dot(BA,BC) / (|BA|*|BC|))
        Returns degrees: 180° = straight, 0° = fully bent.
        """
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def compute_angles(self, landmarks):
        """Compute all 8 joint angles from one frame. Returns dict of angle_name → degrees."""
        angles = {}
        for name, (a_name, b_name, c_name) in self.ANGLE_DEFINITIONS.items():
            # Use 3D coordinates (x, y, z) instead of 2D for accurate biomechanics
            # preventing perspective distortion when hip drops below knee in 2D plane
            a = landmarks[self.LANDMARKS[a_name], :3]
            b = landmarks[self.LANDMARKS[b_name], :3]
            c = landmarks[self.LANDMARKS[c_name], :3]
            angles[name] = self.calculate_angle(a, b, c)
        return angles

    def compute_normalized_coords(self, landmarks):
        """Normalize coordinates relative to hip center, scaled by torso length."""
        left_hip = landmarks[self.LANDMARKS['left_hip'], :2]
        right_hip = landmarks[self.LANDMARKS['right_hip'], :2]
        hip_center = (left_hip + right_hip) / 2

        left_shoulder = landmarks[self.LANDMARKS['left_shoulder'], :2]
        right_shoulder = landmarks[self.LANDMARKS['right_shoulder'], :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center) + 1e-8

        normalized = []
        for idx in self.KEY_LANDMARK_INDICES:
            norm_x = (landmarks[idx, 0] - hip_center[0]) / torso_length
            norm_y = (landmarks[idx, 1] - hip_center[1]) / torso_length
            normalized.extend([norm_x, norm_y])
        return np.array(normalized)

    def compute_velocities(self, current, previous):
        """Compute frame-to-frame displacement of key joints."""
        velocities = []
        for name in self.VELOCITY_JOINTS:
            idx = self.LANDMARKS[name]
            dx = current[idx, 0] - previous[idx, 0]
            dy = current[idx, 1] - previous[idx, 1]
            velocities.extend([dx, dy])
        return np.array(velocities)

    def extract_features_sequence(self, keypoint_sequence):
        """
        Main method: extract features from full video keypoint sequence.

        Args:
            keypoint_sequence: (num_frames, 33, 4) from KeypointExtractor
        Returns:
            features: (num_frames, 84) feature matrix for LSTM
            angle_names: list of angle names (for auto-labeler)
        """
        num_frames = keypoint_sequence.shape[0]
        all_features = []
        
        # Create a mutable copy to freeze occluded joints
        seq_clean = keypoint_sequence.copy()

        for i in range(num_frames):
            frame = seq_clean[i]
            
            # Visibility freezing: If confidence < 0.5, use the last known good position
            if i > 0:
                for j in range(33):
                    if frame[j, 3] < 0.5:
                        frame[j] = seq_clean[i-1, j]
                        
            angles = list(self.compute_angles(frame).values())
            coords = self.compute_normalized_coords(frame)
            velocities = (self.compute_velocities(frame, seq_clean[i-1])
                         if i > 0 else np.zeros(self.num_velocity_features))

            all_features.append(np.concatenate([angles, coords, velocities]))

        return np.array(all_features), list(self.ANGLE_DEFINITIONS.keys())

    def get_primary_angle_for_exercise(self, exercise_type, features, angle_names):
        """Get the main joint angle signal for rep detection."""
        primary = {
            'squat': 'left_knee',
            'pushup': 'left_elbow',
            'pullup': 'left_elbow',
            'plank': 'left_hip',
        }
        idx = angle_names.index(primary.get(exercise_type, 'left_knee'))
        return features[:, idx]


if __name__ == "__main__":
    fe = FeatureEngineer()
    print(f"Features per frame: {fe.total_features}")
    print(f"  Angles: {fe.num_angle_features}, Coords: {fe.num_coord_features}, "
          f"Velocities: {fe.num_velocity_features}")
    print("✅ Feature engineer OK")
