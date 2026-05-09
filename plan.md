Plan
Here's your execution plan — no more design, just doing:
🚀 What To Do Right Now (Tonight, May 8)
Both of You:
1. Download the Kaggle dataset → place it in subfolders by exercise name
2. Set up Python 3.12 venv and install dependencies:
2. Set up Python 3.12 venv and install dependencies:
data/kaggle_exercise_videos/ python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
3. Push/pull the repo so you're both on the same codebase
with
🗓
Day 1 — May 9 (Friday)
Shaheer (You):
Order Task Command Expected Output
1 Run
process_dataset.py
on the Kaggle videos
python process_dataset.py --
data_dir
data/kaggle_exercise_videos
.npy + .json
files in
data/processed/
2 Spot-check results —
open a few
_labels.json files
and verify rep counts
make sense
Manual review If all reps = 0, the
peak detection
thresholds need
tuning
3 Git push the
data/processed/
folder
git add data/processed/ &&
git commit && git push
Abdul Rehman can
now pull
Order Task Command Expected Output
4 Build the real-time
demo — test
real_time_system.py
with your webcam
(without LSTM first,
just skeleton)
python -m
integration.real_time_system
--model models/lstm_best.pth
⚠ This will crash
until the LSTM is
trained — that's
fine, test
MediaPipe
skeleton only for
now
5 (If time) Start COCO
2017 val evaluation for
the CNN benchmark
numbers
Download COCO val2017, write
eval script
OKS/mAP
numbers for slides
Abdul Rehman:
LSTM layer
Order Task Command
1 Git pull —
get the
processed
.npy and
.json files
Shaheer
pushed
git pull
2 Verify data
looks right
python -c "import numpy as np;
d=np.load('data/processed/squat/VIDEO_NAME_features.npy');
print(d.shape)"
3 Train the
LSTM on
Colab T4
Upload repo to Colab, run: python -m lstm_module.train --
data_dir data/processed --epochs 100
4 Check
results —
are rep
accuracy
and form
accuracy
reasonable?
Read terminal output
5 If results
are bad →
tune: lower
lr to
0.0005,
increase
dropout to
0.4, or
simplify to 1
LSTM layer
Edit model.py or pass args
Order Task Command
6 Push the
trained
model
weights +
results
git push (or share lstm_best.pth via Drive if too large for Git)
🗓
Day 2 — May 10 (Saturday)
Shaheer:
Order Task
1 Pull Abdul Rehman's trained model
2 3 with the overlay)
4 Fix any bugs in the real-time system
Run the full real-time demo with the trained LSTM — test all 3 exercises
Record a 2-3 min demo video (screen record yourself doing squats/pushups/curls
Abdul Rehman:
Order Task
1 Run the 3-way evaluation: python -m integration.evaluate --data_dir
data/processed --model models/lstm_best.pth
2 Screenshot/save the comparison table (CNN-only vs LSTM-only vs CNN+LSTM)
3 Generate training loss curves (plot from training_results.json)
Both Together (Evening):
Order Task
1Build presentation slides — include: architecture diagram, comparison table,
demo video, biomechanical label definitions
2 Practice viva Q&A
— quiz each other on the other person's module
3 Make sure the live demo works on the presentation laptop
🗓
Day 3 — May 11 (Sunday): Viva Day
Final dry run morning
Test live demo on the actual presentation machine
Present
⚠ Things That Will Probably Break (and what to do)
Problem Fix
Kaggle folder names don't match code
expectations (e.g. Push Up vs pushup)
Edit the EXERCISE_NAME_MAP dict in
process_dataset.py to add the actual folder
names
All reps detected as 0 Lower the prominence parameter in
auto_labeler.py from 15 to 10 or 8
All form labels are 1 (good) or all 0 (bad) Adjust angle thresholds in auto_labeler.py
— check actual angle ranges in your data first
LSTM loss doesn't decrease Lower learning rate to shapes are correct
0.0005, check that data
real_time_system.py crashes It needs the trained model file — make sure
models/lstm_best.pth exists
MediaPipe import error pip install mediapipe — may need
specific version for Python 3.12
The critical path is: Shaheer processes videos → pushes data → Abdul Rehman trains
LSTM → pushes model → Shaheer runs live demo. Everything else is parallel. Go build it. 💪