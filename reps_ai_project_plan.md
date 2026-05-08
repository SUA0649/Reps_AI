# Reps AI: Project Development Plan
## Team: Shaheer Uddin Ahmed & Abdul Rehman Zuberi

---

## 1. COMPONENT DIVISION STRATEGY

### **Member 1: CNN & Spatial Processing Pipeline**
**Core Responsibility: Spatial Feature Extraction**

#### Primary Components:
- **Custom CNN Architecture Design & Training**
  - Design lightweight CNN from scratch for 17 keypoint detection
  - Implement data augmentation pipeline (horizontal flip, brightness/contrast jitter, random occlusion)
  - Train on COCO-Pose 2017 (200k+ images) and MPII (25k images)
  - Optimize for real-time inference

- **Video Input & Preprocessing Module**
  - OpenCV webcam capture at 30 FPS
  - Frame preprocessing and normalization
  - Real-time frame buffering system

- **Keypoint Extraction & Validation**
  - Post-processing of CNN outputs
  - Keypoint confidence scoring
  - Handling partial occlusions

#### Technical Learning Outcomes:
- CNN architecture design (convolution layers, pooling, batch normalization)
- Transfer learning concepts (COCO → Exercise-specific fine-tuning)
- Image preprocessing and augmentation
- Computer vision fundamentals

---

### **Member 2: LSTM & Temporal Analysis Pipeline**
**Core Responsibility: Temporal Sequence Modeling**

#### Primary Components:
- **LSTM Architecture Design & Training**
  - Design LSTM network for temporal sequence modeling
  - Dual-head architecture: (1) Rep counting, (2) Form classification
  - Train on sequential keypoint data from exercise datasets

- **Fatigue Prediction System**
  - LSTM-based velocity and smoothness analysis
  - Fatigue score calculation from trajectory patterns
  - Warning threshold implementation

- **Form Analysis & Feedback Engine**
  - Temporal pattern recognition (concentric/eccentric phases)
  - Form quality classification (safe vs. dangerous reps)
  - Real-time corrective feedback generation

#### Technical Learning Outcomes:
- LSTM/RNN architectures (gates, cell states, hidden states)
- Sequence-to-sequence modeling
- Temporal feature engineering
- Multi-task learning (rep counting + form classification)

---

## 2. SHARED RESPONSIBILITIES

### **Integration & Testing** (Both Members)
- CNN output → LSTM input pipeline integration
- End-to-end testing with live webcam
- Performance optimization (latency reduction)

### **Dataset Preparation** (Collaborative)
- Download and organize COCO-Pose, MPII, Kaggle datasets
- Create train/validation/test splits
- Generate synthetic temporal sequences for LSTM training

### **Optional Dashboard** (Time Permitting - Divide)
- **Member 1**: FastAPI backend, video clipping logic
- **Member 2**: LLM integration for coaching summaries, Supabase database

---

## 3. DEVELOPMENT PHASES & MILESTONES

### **Phase 1: Foundation & Data Preparation (Week 1-2)**

#### Milestone 1.1: Environment Setup (Both)
- [ ] Set up Python environment with TensorFlow/PyTorch
- [ ] Install OpenCV, NumPy, Matplotlib, etc.
- [ ] Version control setup (Git repository)
- [ ] Download datasets (COCO-Pose, MPII, Kaggle)

#### Milestone 1.2: Data Pipeline (Both - Parallel Work)
- [ ] **Member 1**: COCO/MPII data loaders for CNN training
- [ ] **Member 2**: Sequential data generator for LSTM training
- [ ] Verify data shapes and annotations

**Deliverable**: Functional data loaders for both CNN and LSTM

---

### **Phase 2: Core Model Development (Week 3-5)**

#### Milestone 2.1: CNN Pose Estimation (Member 1)
- [ ] Design CNN architecture (document layers, filters, activations)
- [ ] Implement training loop with loss function (MSE for keypoint regression)
- [ ] Train on COCO-Pose for basic pose estimation
- [ ] Fine-tune on exercise-specific data
- [ ] Achieve >80% keypoint detection accuracy

**Deliverable**: Trained CNN model (.h5/.pth file) + inference script

#### Milestone 2.2: LSTM Sequence Modeling (Member 2)
- [ ] Design dual-head LSTM architecture
- [ ] Implement training loop for rep counting
- [ ] Implement training loop for form classification
- [ ] Generate synthetic temporal sequences for training
- [ ] Achieve >85% rep counting accuracy, >75% form classification

**Deliverable**: Trained LSTM model (.h5/.pth file) + inference script

---

### **Phase 3: Integration & Real-Time System (Week 6-7)**

#### Milestone 3.1: Pipeline Integration (Both)
- [ ] Connect CNN output → LSTM input
- [ ] Implement rolling window for LSTM sequence buffering
- [ ] Real-time webcam → CNN → LSTM → feedback loop
- [ ] Test with live exercises (squats, push-ups, etc.)

#### Milestone 3.2: Fatigue Prediction (Member 2)
- [ ] Implement velocity and smoothness calculation from keypoint trajectories
- [ ] Train fatigue prediction model
- [ ] Integrate into main LSTM pipeline

**Deliverable**: Working real-time system with webcam input

---

### **Phase 4: Polish & Optional Features (Week 8)**

#### Milestone 4.1: UI/Feedback Overlay (Member 1)
- [ ] Real-time feedback overlay on video feed
- [ ] Rep counter display
- [ ] Form quality indicator
- [ ] Fatigue warning system

#### Milestone 4.2: Optional Dashboard (If Time Permits)
- [ ] **Member 1**: FastAPI backend + video clipping
- [ ] **Member 2**: LLM coaching summary + Supabase integration
- [ ] Front-end dashboard for fail-clip review

**Deliverable**: Polished application with user interface

---

## 4. WEEKLY SYNC POINTS

### **Bi-Weekly Integration Sessions**
- **Week 2**: Verify data pipelines work together
- **Week 4**: Test CNN outputs with LSTM inputs (mock integration)
- **Week 6**: Full pipeline integration testing
- **Week 8**: Final debugging and demo preparation

### **Daily Stand-ups (Async)**
- Share progress updates via WhatsApp/Slack
- Flag blockers early
- Share code snippets and issues

---

## 5. DELIVERABLES CHECKLIST

### **Core Deliverables (REQUIRED)**
- [ ] Custom CNN model for 17-keypoint pose estimation
- [ ] Custom LSTM model for rep counting + form analysis
- [ ] Real-time webcam processing system
- [ ] Fatigue prediction feature
- [ ] Dataset preparation documentation
- [ ] Model training logs and performance metrics
- [ ] Final demo video (2-3 minutes)
- [ ] Project report (methodology, results, challenges)

### **Optional Deliverables (Bonus)**
- [ ] Fail-clip dashboard with FastAPI backend
- [ ] LLM-based coaching summary
- [ ] Cloud storage integration
- [ ] Web-based front-end

---

## 6. RISK MITIGATION

### **Backup Plan: Pre-trained Models**
*As noted in proposal: "we can have the ability to use pre-trained or prebuilt models of either CNN or LSTMs not both"*

**If CNN training fails:**
- Member 1 switches to pre-trained MoveNet/YOLO for pose estimation
- Member 2 MUST build custom LSTM (non-negotiable for learning)

**If LSTM training fails:**
- Member 2 switches to rule-based heuristics for rep counting
- Member 1 MUST build custom CNN (non-negotiable for learning)

### **Timeline Buffers**
- Phase 2 has 3 weeks for model development (most critical)
- Phase 4 is flexible (optional features can be dropped)
- Week 7 is integration buffer if Phase 2 overruns

---

## 7. COMMUNICATION & COLLABORATION

### **Code Repository Structure**
```
reps-ai/
├── cnn_module/              # Member 1's work
│   ├── model.py
│   ├── train.py
│   ├── data_loader.py
│   └── inference.py
├── lstm_module/             # Member 2's work
│   ├── model.py
│   ├── train.py
│   ├── sequence_generator.py
│   └── inference.py
├── integration/             # Shared work
│   ├── pipeline.py
│   └── real_time_system.py
├── data/                    # Datasets
├── models/                  # Saved models
├── notebooks/               # Jupyter notebooks for experimentation
└── dashboard/               # Optional dashboard code
```

### **Documentation Requirements**
- Each member documents their module with README.md
- Code comments for complex logic
- Training logs (loss curves, accuracy metrics)
- Architecture diagrams (use draw.io or similar)

---

## 8. LEARNING OBJECTIVES VERIFICATION

### **Member 1 (CNN Focus)**
✅ Understands convolution operations, pooling, feature maps  
✅ Implements backpropagation through CNN layers  
✅ Applies data augmentation for robustness  
✅ Optimizes for real-time inference  

### **Member 2 (LSTM Focus)**
✅ Understands LSTM gates (forget, input, output)  
✅ Implements sequence modeling and temporal dependencies  
✅ Handles variable-length sequences  
✅ Implements multi-task learning (dual heads)  

### **Both Members**
✅ End-to-end deep learning pipeline (data → training → inference)  
✅ Model evaluation and performance metrics  
✅ Real-time system integration  
✅ Debugging deep learning models  

---

## 9. SUCCESS CRITERIA

### **Minimum Viable Product (MVP)**
- [x] CNN detects 17 keypoints with >70% accuracy
- [x] LSTM counts reps with >80% accuracy
- [x] System runs in real-time (>15 FPS)
- [x] Basic form feedback (binary: good/bad)

### **Target Product**
- [x] CNN detects 17 keypoints with >85% accuracy
- [x] LSTM counts reps with >90% accuracy
- [x] Form classification with >80% accuracy
- [x] Fatigue prediction implemented
- [x] Real-time overlay with feedback

### **Stretch Goals**
- [x] Dashboard with fail-clip review
- [x] LLM coaching summaries
- [x] Support for 3+ different exercises

---

## 10. FINAL SUBMISSION CHECKLIST

### **Code Submission**
- [ ] Clean, commented codebase
- [ ] Requirements.txt for dependencies
- [ ] Model weights (.h5/.pth files)
- [ ] Inference script for demo

### **Documentation**
- [ ] Project report (PDF, 8-10 pages)
- [ ] Architecture diagrams
- [ ] Training methodology and hyperparameters
- [ ] Results and performance analysis
- [ ] Challenges faced and solutions

### **Demo**
- [ ] 2-3 minute video demonstration
- [ ] Live demo during presentation (if required)
- [ ] Show edge cases (occlusion, different angles, fatigue)

---

## NEXT STEPS

1. **Week 1 Kick-off**:
   - Schedule first sync meeting
   - Assign Member 1 vs Member 2 based on preferences
   - Set up Git repository
   - Download datasets

2. **Establish Communication**:
   - Create WhatsApp/Slack group
   - Set up shared Google Drive for papers/resources
   - Agree on coding standards (PEP 8, etc.)

3. **Start Coding**:
   - Member 1: Begin CNN architecture design
   - Member 2: Begin LSTM architecture design
   - Both: Data pipeline setup

---

**Let's build something amazing! 💪🏋️‍♂️🤖**
