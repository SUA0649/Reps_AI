"""
Real-Time Workout Analysis System (Shaheer)
=============================================
Webcam → MediaPipe → Features → LSTM → Overlay Display

This is the DEMO system shown during the viva.
It runs the full pipeline in real-time at 25-30 FPS.

Usage:
  python -m integration.real_time_system --model models/lstm_best.pth --exercise squat
"""

import cv2
import numpy as np
import torch
import argparse
import time
from collections import deque

from cnn_module.keypoint_extractor import KeypointExtractor
from cnn_module.feature_engineer import FeatureEngineer
from lstm_module.model import ExerciseLSTM


class RealTimeSystem:
    """Real-time exercise analysis with webcam input."""

    def __init__(self, model_path, exercise_type='squat',
                 window_size=60, device=None):
        """
        Args:
            model_path: path to trained LSTM weights (.pth)
            exercise_type: which exercise to analyze
            window_size: number of frames to buffer for LSTM
            device: 'cuda' or 'cpu'
        """
        self.exercise_type = exercise_type
        self.window_size = window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize modules
        self.extractor = KeypointExtractor(model_complexity=1)
        self.feature_eng = FeatureEngineer()

        # Load trained LSTM model
        self.model = ExerciseLSTM(input_size=84).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Rolling buffer for frame features
        self.feature_buffer = deque(maxlen=window_size)

        # State tracking
        self.total_reps = 0
        self.current_form = "N/A"
        self.form_color = (200, 200, 200)
        self.prev_rep_pred = 0
        self.fps_tracker = deque(maxlen=30)

    def process_frame(self, frame):
        """
        Process a single frame through the full pipeline.

        Returns:
            annotated_frame: frame with skeleton and info overlay
        """
        start_time = time.time()

        # Step 1: Extract keypoints
        landmarks = self.extractor.extract_from_frame(frame)

        if landmarks is None:
            return self._draw_no_person(frame)

        # Step 2: Draw skeleton
        annotated = self.extractor.draw_skeleton(frame, landmarks)

        # Step 3: Compute features
        angles = self.feature_eng.compute_angles(landmarks)
        coords = self.feature_eng.compute_normalized_coords(landmarks)

        if self.feature_buffer:
            prev_landmarks = self._last_landmarks
            velocities = self.feature_eng.compute_velocities(landmarks, prev_landmarks)
        else:
            velocities = np.zeros(self.feature_eng.num_velocity_features)

        self._last_landmarks = landmarks.copy()

        frame_features = np.concatenate([
            list(angles.values()), coords, velocities
        ])
        self.feature_buffer.append(frame_features)

        # Step 4: Run LSTM when buffer is full
        if len(self.feature_buffer) >= self.window_size:
            self._run_lstm_inference()

        # Step 5: Draw info overlay
        annotated = self._draw_overlay(annotated, angles)

        # Track FPS
        elapsed = time.time() - start_time
        self.fps_tracker.append(elapsed)

        return annotated

    def _run_lstm_inference(self):
        """Run LSTM on the current feature buffer."""
        # Convert buffer to tensor
        sequence = np.array(list(self.feature_buffer))  # (window_size, 84)
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 60, 84)

        with torch.no_grad():
            rep_pred, form_pred = self.model(x)

        rep_count = rep_pred.item()
        form_prob = form_pred.item()

        # Accumulate rep count (detect when a new rep is completed)
        rounded_reps = round(rep_count)
        if rounded_reps > self.prev_rep_pred and rounded_reps > 0:
            self.total_reps += (rounded_reps - self.prev_rep_pred)
        self.prev_rep_pred = rounded_reps

        # Update form display
        if form_prob > 0.7:
            self.current_form = "GOOD FORM"
            self.form_color = (0, 200, 0)     # Green
        elif form_prob > 0.4:
            self.current_form = "CHECK FORM"
            self.form_color = (0, 200, 200)   # Yellow
        else:
            self.current_form = "BAD FORM"
            self.form_color = (0, 0, 200)     # Red

    def _draw_overlay(self, frame, angles):
        """Draw info panel on the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Exercise name
        cv2.putText(frame, f"Exercise: {self.exercise_type.upper()}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Rep counter
        cv2.putText(frame, f"Reps: {self.total_reps}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Form quality
        cv2.putText(frame, self.current_form,
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.form_color, 2)

        # Primary joint angle
        primary_angle_map = {
            'squat': 'left_knee',
            'pushup': 'left_elbow',
            'hammer_curl': 'left_elbow',
        }
        primary = primary_angle_map.get(self.exercise_type, 'left_knee')
        angle_val = angles.get(primary, 0)
        cv2.putText(frame, f"Angle: {angle_val:.0f} deg",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # FPS
        if self.fps_tracker:
            avg_fps = 1.0 / (np.mean(self.fps_tracker) + 1e-8)
            cv2.putText(frame, f"FPS: {avg_fps:.0f}",
                        (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Buffer status
        buf_pct = len(self.feature_buffer) / self.window_size
        bar_w = 150
        cv2.rectangle(frame, (w - bar_w - 20, 20), (w - 20, 35), (100, 100, 100), -1)
        cv2.rectangle(frame, (w - bar_w - 20, 20),
                      (w - bar_w - 20 + int(bar_w * buf_pct), 35), (0, 255, 0), -1)

        return frame

    def _draw_no_person(self, frame):
        """Show message when no person is detected."""
        cv2.putText(frame, "No person detected",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def run(self):
        """Main loop: capture webcam → process → display."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return

        print(f"🏋️ Real-time {self.exercise_type} analysis started!")
        print("Press 'q' to quit, 'r' to reset rep count, 'e' to switch exercise")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            annotated = self.process_frame(frame)
            cv2.imshow('Reps AI - Real-Time Workout Analysis', annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_reps = 0
                self.prev_rep_pred = 0
                print("🔄 Rep count reset")
            elif key == ord('e'):
                exercises = ['squat', 'pushup', 'hammer_curl']
                idx = exercises.index(self.exercise_type)
                self.exercise_type = exercises[(idx + 1) % 3]
                self.total_reps = 0
                self.prev_rep_pred = 0
                self.feature_buffer.clear()
                print(f"🔄 Switched to {self.exercise_type}")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        print("✅ Session ended")

    def cleanup(self):
        """Release resources."""
        self.extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time workout analysis')
    parser.add_argument('--model', type=str, default='models/lstm_best.pth',
                        help='Path to trained LSTM model')
    parser.add_argument('--exercise', type=str, default='squat',
                        choices=['squat', 'pushup', 'hammer_curl'])
    args = parser.parse_args()

    system = RealTimeSystem(args.model, args.exercise)
    system.run()
