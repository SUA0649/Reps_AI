"""
Keypoint Extraction Module using MediaPipe Pose (Shaheer)
=========================================================
Wraps Google's MediaPipe PoseLandmarker (Tasks API) to extract 33 body
landmarks from video frames. Each landmark has (x, y, z, visibility).



MediaPipe Pose Landmarks (33 total):
  0: nose
  1-10: face landmarks (eyes, ears, mouth)
  11: left_shoulder,  12: right_shoulder
  13: left_elbow,     14: right_elbow
  15: left_wrist,     16: right_wrist
  17-22: hand landmarks (pinky, index, thumb)
  23: left_hip,       24: right_hip
  25: left_knee,      26: right_knee
  27: left_ankle,     28: right_ankle
  29-32: foot landmarks (heel, foot_index)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# New Tasks API imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class KeypointExtractor:
    """Extract pose keypoints from video frames using MediaPipe PoseLandmarker."""

    # Map MediaPipe's 33 landmarks → COCO's 17 keypoints (for benchmark evaluation)
    MEDIAPIPE_TO_COCO = {
        0: 'nose',
        2: 'left_eye',     5: 'right_eye',
        7: 'left_ear',     8: 'right_ear',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow',  14: 'right_elbow',
        15: 'left_wrist',  16: 'right_wrist',
        23: 'left_hip',    24: 'right_hip',
        25: 'left_knee',   26: 'right_knee',
        27: 'left_ankle',  28: 'right_ankle',
    }

    # Default model path (download to models/ directory)
    DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / 'models' / 'pose_landmarker_heavy.task')

    def __init__(self, model_path=None, num_poses=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe PoseLandmarker.

        Args:
            model_path: path to .task model file. If None, uses default location.
            num_poses: max number of people to detect (1 for our use case)
            min_detection_confidence: threshold for initial person detection.
            min_tracking_confidence: threshold for landmark tracking between frames.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download it with:\n"
                "  curl -L -o models/pose_landmarker_heavy.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                "pose_landmarker_heavy.task"
            )

        # We'll create landmarker instances per-use because VIDEO mode requires
        # monotonically increasing timestamps
        self._num_poses = num_poses
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def _create_landmarker(self, mode='IMAGE'):
        """Create a PoseLandmarker instance for the given mode."""
        running_mode = {
            'IMAGE': VisionRunningMode.IMAGE,
            'VIDEO': VisionRunningMode.VIDEO,
        }[mode]

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=running_mode,
            num_poses=self._num_poses,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return PoseLandmarker.create_from_options(options)

    def _result_to_numpy(self, result):
        """
        Convert PoseLandmarkerResult to numpy array.

        Args:
            result: PoseLandmarkerResult from the landmarker

        Returns:
            landmarks: (33, 4) numpy array of (x, y, z, visibility)
                       Returns None if no person detected.
        """
        if not result.pose_landmarks:
            return None

        # Take the first detected person
        pose = result.pose_landmarks[0]

        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose
        ])  # Shape: (33, 4)

        return landmarks

    def extract_from_frame(self, frame):
        """
        Extract 33 landmarks from a single BGR frame (IMAGE mode).

        Args:
            frame: BGR image (H, W, 3) from OpenCV

        Returns:
            landmarks: numpy array shape (33, 4) → (x, y, z, visibility)
                       Returns None if no person detected.
        """
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Use IMAGE mode for single-frame detection
        with self._create_landmarker('IMAGE') as landmarker:
            result = landmarker.detect(mp_image)

        return self._result_to_numpy(result)

    def extract_from_video(self, video_path, target_fps=30):
        """
        Extract keypoint sequences from an entire video file.

        Args:
            video_path: Path to .mp4 video file
            target_fps: Standardize all videos to this FPS (default 30)

        Returns:
            keypoints: numpy array of shape (num_frames, 33, 4)
            metadata: dict with video info
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame sampling to achieve target FPS
        frame_interval = max(1, round(original_fps / target_fps)) if original_fps > 0 else 1

        all_keypoints = []
        frame_idx = 0
        timestamp_ms = 0
        frame_duration_ms = int(1000 / target_fps)

        # Use VIDEO mode for frame-to-frame tracking
        with self._create_landmarker('VIDEO') as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    landmarks = self._result_to_numpy(result)

                    if landmarks is not None:
                        all_keypoints.append(landmarks)
                    elif all_keypoints:
                        # Brief occlusion: reuse last known keypoints
                        all_keypoints.append(all_keypoints[-1].copy())

                    timestamp_ms += frame_duration_ms

                frame_idx += 1

        cap.release()

        if not all_keypoints:
            raise ValueError(f"No keypoints detected in video: {video_path}")

        keypoints = np.array(all_keypoints)  # (num_frames, 33, 4)

        metadata = {
            'video_path': str(video_path),
            'original_fps': original_fps,
            'target_fps': target_fps,
            'total_original_frames': total_frames,
            'keypoint_frames': len(all_keypoints),
        }

        return keypoints, metadata

    def get_coco_keypoints(self, landmarks):
        """
        Map MediaPipe 33 landmarks → COCO 17 keypoints for benchmark evaluation.

        Args:
            landmarks: (33, 4) array of MediaPipe landmarks

        Returns:
            coco_kps: (17, 3) array of (x, y, confidence)
        """
        coco_indices = sorted(self.MEDIAPIPE_TO_COCO.keys())
        coco_kps = np.zeros((17, 3))

        for i, mp_idx in enumerate(coco_indices):
            coco_kps[i, 0] = landmarks[mp_idx, 0]   # x
            coco_kps[i, 1] = landmarks[mp_idx, 1]   # y
            coco_kps[i, 2] = landmarks[mp_idx, 3]   # visibility → confidence

        return coco_kps

    def draw_skeleton(self, frame, landmarks):
        """
        Draw pose skeleton overlay on a frame for visualization.

        Args:
            frame: BGR image to draw on (will be copied)
            landmarks: (33, 4) array of landmarks

        Returns:
            annotated_frame: copy of frame with skeleton drawn
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Skeleton connections: pairs of landmark indices
        connections = [
            (11, 13), (13, 15),   # Left arm
            (12, 14), (14, 16),   # Right arm
            (11, 12),             # Shoulder line
            (11, 23), (12, 24),   # Torso sides
            (23, 24),             # Hip line
            (23, 25), (25, 27),   # Left leg
            (24, 26), (26, 28),   # Right leg
        ]

        for start_idx, end_idx in connections:
            start = (int(landmarks[start_idx, 0] * w), int(landmarks[start_idx, 1] * h))
            end = (int(landmarks[end_idx, 0] * w), int(landmarks[end_idx, 1] * h))
            cv2.line(annotated, start, end, (0, 255, 0), 2)

        for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if landmarks[i, 3] > 0.5:
                cx, cy = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        return annotated

    def close(self):
        """No persistent resources to release in Tasks API."""
        pass


# ============================================================
# Quick test: Run from project root with:
#   python -m cnn_module.keypoint_extractor [optional_video_path]
# ============================================================
if __name__ == "__main__":
    import sys

    extractor = KeypointExtractor()

    if len(sys.argv) > 1:
        # Test on a video file
        video_path = sys.argv[1]
        print(f"Processing video: {video_path}")
        keypoints, meta = extractor.extract_from_video(video_path)
        print(f"Extracted {keypoints.shape[0]} frames, shape: {keypoints.shape}")
        print(f"Metadata: {meta}")
    else:
        # Test on webcam (single frame)
        print("Testing with webcam (press 'q' to quit)...")
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = extractor.extract_from_frame(frame)

            if landmarks is not None:
                frame = extractor.draw_skeleton(frame, landmarks)
                cv2.putText(frame, f"Landmarks: {landmarks.shape[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Keypoint Extractor Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    extractor.close()
    print("✅ Keypoint extractor test complete!")
