# 🏋️ Reps AI — AI-Powered Real-Time Workout Analysis

> **Semester Project — Deep Learning / Computer Vision**
>
> An AI-powered workout coaching system that uses custom-trained deep learning architectures to analyze exercise form through a standard webcam, count repetitions automatically, and deliver real-time corrective feedback. No specialized equipment required.

---

## 👥 Team

| Member | Role | Responsibilities |
|--------|------|------------------|
| **Shaheer Uddin Ahmed** | CNN / Spatial Pipeline Lead | MediaPipe pose estimation integration, keypoint extraction (`keypoint_extractor.py`), biomechanical feature engineering (`feature_engineer.py`), CNN-only baseline evaluation (`cnn_baseline.py`), dataset processing pipeline (`process_dataset.py`), real-time system (`real_time_system.py`) |
| **Abdul Rehman** | LSTM / Temporal Pipeline Lead | Dual-head LSTM architecture design (`model.py`), PyTorch dataset & sliding window creation (`dataset.py`), automatic labeling system (`auto_labeler.py`), training pipeline with early stopping (`train.py`), LSTM inference module (`inference.py`), LSTM-only baseline (`baseline_lstm_only.py`) |
| **Both** | Integration & Evaluation | Three-way model comparison (`evaluate.py`), MPII/COCO/Kaggle benchmark scripts, real-time system refinement, plank rule-based system, pull-up optimizations, architecture documentation |

---

## 🎯 Supported Exercises

| Exercise | Detection Method | Metric |
|----------|-----------------|--------|
| **Squat** | LSTM Form + Angle State Machine | Rep Count + Form Quality |
| **Push-up** | LSTM Form + Angle State Machine | Rep Count + Form Quality |
| **Pull-up** | LSTM Form + Dual-Elbow Averaging | Rep Count + Form Quality |
| **Plank** | Rule-Based (Hip Angle > 150°) | Hold Duration Timer + Form Quality |

---

## 🏗️ System Architecture

Reps AI follows a **two-stage Spatial-to-Temporal pipeline**:

```
📹 Webcam / Video
       ↓
┌──────────────────────────────────┐
│  Stage 1: CNN (Spatial)          │
│  MediaPipe PoseLandmarker        │
│  → 33 body landmarks (x,y,z,v)  │
│  → Visibility Filtering          │
│  → FeatureEngineer (84 features) │
│    • 8 Joint Angles (3D)         │
│    • 52 Normalized Coordinates   │
│    • 24 Velocity Features        │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  Stage 2: LSTM (Temporal)        │
│  ExerciseLSTM (163K params)      │
│  → 60-frame sliding window       │
│  → LSTM Layer 1 (hidden=128)     │
│  → LSTM Layer 2 (hidden=64)      │
│  → Rep Head (regression)         │
│  → Form Head (sigmoid → 0-1)     │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│  Output                          │
│  • Skeleton Overlay              │
│  • Rep Count / Plank Timer       │
│  • Form: GOOD / BAD (+ prob)    │
│  • FPS Counter                   │
└──────────────────────────────────┘
```

---

## 📁 Project Structure

```
Reps_AI/
├── cnn_module/                     # Stage 1 — Spatial Pipeline (Shaheer)
│   ├── keypoint_extractor.py       #   MediaPipe PoseLandmarker wrapper
│   ├── feature_engineer.py         #   84-feature biomechanical extraction
│   └── cnn_baseline.py             #   CNN-only baseline (peak detection)
│
├── lstm_module/                    # Stage 2 — Temporal Pipeline (Abdul Rehman)
│   ├── model.py                    #   Dual-head ExerciseLSTM architecture
│   ├── dataset.py                  #   Sliding window dataset + augmentation
│   ├── auto_labeler.py             #   Automatic rep detection + form labeling
│   ├── train.py                    #   Training with early stopping
│   ├── inference.py                #   Standalone video inference
│   └── baseline_lstm_only.py       #   LSTM-only baseline evaluation
│
├── integration/                    # Combined System (Both)
│   ├── real_time_system.py         #   Live webcam/video demo application
│   ├── evaluate.py                 #   Three-way model comparison
│   ├── evaluate_mpii.py            #   MPII Human Pose benchmark (PCKh@0.5)
│   ├── evaluate_coco.py            #   COCO 2017 Keypoints benchmark (PCK@0.2)
│   ├── evaluate_kaggle.py          #   Kaggle Pose CSV benchmark
│   └── analyze_dataset_labels.py   #   Training data distribution analysis
│
├── models/                         # Model weights
│   ├── pose_landmarker_heavy.task  #   MediaPipe Heavy model
│   ├── pose_landmarker_full.task   #   MediaPipe Full model
│   ├── pose_landmarker_lite.task   #   MediaPipe Lite model
│   └── lstm_best.pth              #   Trained LSTM weights
│
├── data/
│   ├── kaggle_exercise_videos/     #   Raw training videos (per exercise folder)
│   └── processed/                  #   Extracted features + labels (.npy + .json)
│
├── process_dataset.py              # Dataset processing orchestration script
├── requirements.txt                # Python dependencies
├── SYSTEM_DOCUMENTATION.md         # Detailed technical documentation
└── README.md                       # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.12
- macOS (Apple Silicon MPS) / Linux / Windows

### Setup

```bash
# Clone the repository
git clone https://github.com/shaheeruddinahmed/Reps_AI.git
cd Reps_AI

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download MediaPipe Models

```bash
# Heavy model (highest accuracy)
curl -L -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# Full model (balanced)
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# Lite model (fastest)
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

---

## 🚀 Usage

### Real-Time Demo (Webcam)

```bash
# Squat analysis with webcam
python -m integration.real_time_system --exercise squat --full

# Push-up analysis
python -m integration.real_time_system --exercise pushup --full

# Pull-up analysis
python -m integration.real_time_system --exercise pullup --full

# Plank timer
python -m integration.real_time_system --exercise plank --full
```

### Analyze a Video File

```bash
python -m integration.real_time_system --exercise squat --full --input path/to/video.mp4
```

### Controls
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset rep count / plank timer |
| `e` | Cycle to next exercise |

---

## 🔧 Training Pipeline

### Step 1: Process Dataset

Place your exercise videos in `data/kaggle_exercise_videos/` organized by exercise folder (e.g., `squat/`, `pushup/`, `pullup/`, `plank/`).

```bash
python process_dataset.py --data_dir data/kaggle_exercise_videos --output_dir data/processed --full
```

### Step 2: Train LSTM

```bash
python -m lstm_module.train --data_dir data/processed --output_dir models
```

### Step 3: Evaluate

```bash
# Three-way comparison (CNN-Only vs LSTM-Only vs CNN+LSTM)
python -m integration.evaluate --data_dir data/processed --model models/lstm_best.pth

# Dataset label distribution
python -m integration.analyze_dataset_labels
```

---

## 📊 Results

### LSTM Training Performance

| Metric | Value |
|--------|-------|
| Rep Count MAE | 0.25 |
| Rep Count Accuracy | 75.3% |
| Form Classification Accuracy | **80.4%** |
| Epochs Trained | 56 (early stopped) |

### Dataset Composition

| Exercise | Videos | Reps |
|----------|--------|------|
| Squat | 23 | 50 |
| Push-up | 56 | 173 |
| Pull-up | 14 | ~50 |
| Plank | 5 | N/A (rule-based) |

---

## 🧠 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **MediaPipe over OpenPose** | Runs at 30+ FPS on CPU, no GPU required for the demo |
| **LSTM over Transformer** | Small dataset (~1000 samples) — Transformer would overfit |
| **Angle state machine for reps** | Deterministic, no warm-up needed, works from frame 1 |
| **Rule-based for planks** | Isometric exercises have no reps — LSTM training produces garbage labels |
| **3D angle calculation** | Prevents perspective distortion when joints move in depth |
| **Visibility filtering** | Freezes occluded joints to prevent hallucination noise |
| **Dual-elbow averaging for pull-ups** | Handles MediaPipe's left/right confusion from rear-view |

---

## 📚 References

- [MediaPipe Pose Landmark Detection](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
- [COCO Keypoint Detection](https://cocodataset.org/#keypoints-2017)
- [BlazePose: On-device Real-time Body Pose Tracking](https://arxiv.org/abs/2006.10204)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
