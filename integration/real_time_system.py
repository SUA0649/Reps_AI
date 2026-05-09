"""
Real-Time Workout Analysis System (Shaheer)
=============================================
Webcam → MediaPipe → Features → LSTM → Overlay Display

Optimized for MacBook:
  - Persistent PoseLandmarker in VIDEO mode (no per-frame model loading)
  - Downscaled processing resolution (640x480)
  - LSTM inference every 15 frames (not every frame)
  - Lightweight model option for slower hardware

Usage:
  python -m integration.real_time_system --model models/lstm_best.pth --exercise squat
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
import argparse
import time
from collections import deque
from pathlib import Path

from cnn_module.feature_engineer import FeatureEngineer
from lstm_module.model import ExerciseLSTM

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Default model paths
POSE_MODEL_HEAVY = str(Path(__file__).parent.parent / 'models' / 'pose_landmarker_heavy.task')
POSE_MODEL_LITE = str(Path(__file__).parent.parent / 'models' / 'pose_landmarker_lite.task')


class RealTimeSystem:
    """Real-time exercise analysis with webcam input, optimized for MacBook."""

    def __init__(self, lstm_model_path, exercise_type='squat',
                 window_size=60, pose_model_path=None, device=None):
        """
        Args:
            lstm_model_path: path to trained LSTM weights (.pth)
            exercise_type: which exercise to analyze
            window_size: number of frames to buffer for LSTM
            pose_model_path: path to MediaPipe .task model file
            device: 'cuda', 'mps', or 'cpu'
        """
        self.exercise_type = exercise_type
        self.window_size = window_size
        self.pose_model_path = pose_model_path or POSE_MODEL_PATH

        # Pick best available device (MPS = Apple Silicon GPU)
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Initialize feature engineer
        self.feature_eng = FeatureEngineer()

        # Load trained LSTM model
        self.model = ExerciseLSTM(input_size=84).to(self.device)
        if Path(lstm_model_path).exists():
            self.model.load_state_dict(
                torch.load(lstm_model_path, map_location=self.device, weights_only=True)
            )
            print(f"✅ LSTM model loaded from {lstm_model_path}")
        else:
            print(f"⚠️ LSTM model not found at {lstm_model_path} — running skeleton-only mode")
        self.model.eval()

        # Rolling buffer for frame features
        self.feature_buffer = deque(maxlen=window_size)

        # State tracking
        self.total_reps = 0
        self.current_form = "Warming up..."
        self.form_color = (200, 200, 200)
        self.prev_rep_pred = 0
        self.fps_times = deque(maxlen=30)
        self._last_landmarks = None

        # LSTM inference throttle
        self._frame_count = 0
        self._lstm_interval = 15

        # Angle-based rep counter (more reliable than LSTM for counting)
        self._angle_history = deque(maxlen=90)  # 3 seconds of angle data
        self._rep_cooldown = 0  # frames to wait after detecting a rep
        self._prev_angle_state = 'up'  # 'up' or 'down' — tracks rep phases

    def _landmarks_to_numpy(self, result):
        """Convert PoseLandmarkerResult to (33, 4) numpy array."""
        if not result.pose_landmarks:
            return None
        pose = result.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose])

    def run(self):
        """Main loop: capture webcam → process → display."""

        # Verify pose model exists
        if not Path(self.pose_model_path).exists():
            print(f"❌ Pose model not found: {self.pose_model_path}")
            print("Download with:")
            print("  curl -L -o models/pose_landmarker_heavy.task \\")
            print("    https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                  "pose_landmarker_heavy.task")
            return

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        # Set camera resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"🏋️ Real-time {self.exercise_type} analysis started!")
        print(f"   Device: {self.device}")
        print("   Press 'q' to quit, 'r' to reset reps, 'e' to switch exercise")

        # Create ONE persistent PoseLandmarker in VIDEO mode
        # This is the key optimization — no per-frame model loading
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            timestamp_ms = 0

            while cap.isOpened():
                frame_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror for natural interaction
                frame = cv2.flip(frame, 1)

                # ---- POSE ESTIMATION (MediaPipe) ----
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += 33  # ~30fps (1000ms / 30)

                landmarks = self._landmarks_to_numpy(result)

                if landmarks is None:
                    cv2.putText(frame, "No person detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Reps AI', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ---- DRAW SKELETON ----
                annotated = self._draw_skeleton(frame, landmarks)

                # ---- COMPUTE FEATURES ----
                angles = self.feature_eng.compute_angles(landmarks)
                coords = self.feature_eng.compute_normalized_coords(landmarks)

                if self._last_landmarks is not None:
                    velocities = self.feature_eng.compute_velocities(
                        landmarks, self._last_landmarks
                    )
                else:
                    velocities = np.zeros(self.feature_eng.num_velocity_features)

                self._last_landmarks = landmarks.copy()

                frame_features = np.concatenate([
                    list(angles.values()), coords, velocities
                ])
                self.feature_buffer.append(frame_features)

                # ---- ANGLE-BASED REP COUNTING (reliable) ----
                primary_map = {'squat': 'left_knee', 'pushup': 'left_elbow',
                               'hammer_curl': 'left_elbow'}
                primary_angle_name = primary_map.get(self.exercise_type, 'left_knee')
                primary_angle = angles.get(primary_angle_name, 180)
                self._angle_history.append(primary_angle)
                self._count_rep_by_angle(primary_angle)

                # ---- LSTM INFERENCE for FORM only (throttled) ----
                self._frame_count += 1
                if (len(self.feature_buffer) >= self.window_size and
                        self._frame_count % self._lstm_interval == 0):
                    self._run_lstm_inference()

                # ---- DRAW OVERLAY ----
                annotated = self._draw_overlay(annotated, angles)

                # ---- FPS tracking ----
                elapsed = time.perf_counter() - frame_start
                self.fps_times.append(elapsed)

                cv2.imshow('Reps AI', annotated)

                # ---- KEY HANDLING ----
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.total_reps = 0
                    self.prev_rep_pred = 0
                    self.feature_buffer.clear()
                    self._last_landmarks = None
                    print("🔄 Rep count reset")
                elif key == ord('e'):
                    exercises = ['squat', 'pushup', 'hammer_curl']
                    idx = exercises.index(self.exercise_type)
                    self.exercise_type = exercises[(idx + 1) % 3]
                    self.total_reps = 0
                    self.prev_rep_pred = 0
                    self.feature_buffer.clear()
                    self._last_landmarks = None
                    print(f"🔄 Switched to {self.exercise_type}")

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Session ended")

    def _count_rep_by_angle(self, current_angle):
        """
        Count reps using angle state machine (up/down transitions).
        Much more reliable than LSTM regression for counting.

        A rep = angle goes below threshold (down) then back above (up).
        """
        # Thresholds per exercise
        thresholds = {
            'squat':       {'down': 120, 'up': 150},  # Knee: <120° = bottom, >150° = standing
            'pushup':      {'down': 110, 'up': 150},  # Elbow: <110° = bottom, >150° = top
            'hammer_curl': {'down': 70,  'up': 130},  # Elbow: <70° = curled, >130° = extended
        }
        t = thresholds.get(self.exercise_type, {'down': 110, 'up': 150})

        if self._rep_cooldown > 0:
            self._rep_cooldown -= 1
            return

        if self._prev_angle_state == 'up' and current_angle < t['down']:
            self._prev_angle_state = 'down'
        elif self._prev_angle_state == 'down' and current_angle > t['up']:
            self._prev_angle_state = 'up'
            self.total_reps += 1
            self._rep_cooldown = 10  # Ignore next 10 frames (~0.3s debounce)
            print(f"\r   Rep #{self.total_reps} detected (angle: {current_angle:.0f}°)", end='')

    def _run_lstm_inference(self):
        """Run LSTM — used for FORM prediction only (rep counting uses angle method)."""
        sequence = np.array(list(self.feature_buffer))
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            rep_pred, form_pred = self.model(x)

        rep_count = rep_pred.item()
        form_prob = form_pred.item()

        # (Rep counting now handled by _count_rep_by_angle — LSTM used for form only)

        # Update form display
        # Thresholds tuned for real-world (model trained on clean Kaggle videos,
        # live conditions are noisier so probabilities sit lower)
        self.current_form_prob = form_prob  # store for overlay display
        if form_prob > 0.5:
            self.current_form = "GOOD FORM"
            self.form_color = (0, 200, 0)     # Green
        elif form_prob > 0.3:
            self.current_form = "CHECK FORM"
            self.form_color = (0, 200, 200)
        else:
            self.current_form = "BAD FORM"
            self.form_color = (0, 0, 200)

    def _draw_skeleton(self, frame, landmarks):
        """Draw pose skeleton on frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (11, 23), (12, 24),  # Torso
            (23, 24),            # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]

        for s, e in connections:
            pt1 = (int(landmarks[s, 0] * w), int(landmarks[s, 1] * h))
            pt2 = (int(landmarks[e, 0] * w), int(landmarks[e, 1] * h))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if landmarks[i, 3] > 0.5:
                cx, cy = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        return annotated

    def _draw_overlay(self, frame, angles):
        """Draw info panel on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Exercise name
        cv2.putText(frame, f"Exercise: {self.exercise_type.upper()}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Rep counter
        cv2.putText(frame, f"Reps: {self.total_reps}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Form quality + raw probability (for debugging/tuning)
        form_prob_display = getattr(self, 'current_form_prob', 0.0)
        cv2.putText(frame, f"{self.current_form} ({form_prob_display:.2f})",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.form_color, 2)

        # Primary joint angle
        primary_map = {'squat': 'left_knee', 'pushup': 'left_elbow',
                       'hammer_curl': 'left_elbow'}
        primary = primary_map.get(self.exercise_type, 'left_knee')
        angle_val = angles.get(primary, 0)
        cv2.putText(frame, f"Angle: {angle_val:.0f} deg",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # FPS
        if self.fps_times:
            avg_fps = 1.0 / (np.mean(self.fps_times) + 1e-8)
            cv2.putText(frame, f"FPS: {avg_fps:.0f}",
                        (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Buffer progress bar
        buf_pct = len(self.feature_buffer) / self.window_size
        bar_w = 150
        cv2.rectangle(frame, (w - bar_w - 20, 20), (w - 20, 35), (100, 100, 100), -1)
        cv2.rectangle(frame, (w - bar_w - 20, 20),
                      (w - bar_w - 20 + int(bar_w * buf_pct), 35), (0, 255, 0), -1)

        return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time workout analysis')
    parser.add_argument('--model', type=str, default='models/lstm_best.pth',
                        help='Path to trained LSTM model')
    parser.add_argument('--exercise', type=str, default='squat',
                        choices=['squat', 'pushup', 'hammer_curl'])
    parser.add_argument('--pose_model', type=str, default=None,
                        help='Path to MediaPipe .task model')
    parser.add_argument('--lite', action='store_true',
                        help='Use lite pose model for better FPS (~15-20 FPS vs ~9 FPS)')
    args = parser.parse_args()

    # Choose pose model
    if args.pose_model:
        pose_path = args.pose_model
    elif args.lite:
        pose_path = POSE_MODEL_LITE
    else:
        pose_path = POSE_MODEL_HEAVY

    system = RealTimeSystem(
        lstm_model_path=args.model,
        exercise_type=args.exercise,
        pose_model_path=pose_path,
    )
    system.run()
