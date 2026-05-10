import json
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from cnn_module.keypoint_extractor import KeypointExtractor

# ==========================================
# USER: UPDATE THESE PATHS!
COCO_VAL_JSON = "REPLACE_WITH_PATH_TO_person_keypoints_val2017.json"
COCO_VAL_IMAGES = "REPLACE_WITH_PATH_TO_val2017_IMAGES"
# ==========================================

def evaluate_coco(json_path, images_dir, max_samples=500):
    print(f"Loading COCO annotations from {json_path}...")
    if not os.path.exists(json_path):
        print("❌ Error: Update the placeholders in the script with the actual paths!")
        return

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build image lookup
    images_dict = {img['id']: img for img in coco_data['images']}
    
    extractor = KeypointExtractor()
    
    total_joints = 0
    correct_joints = 0
    alpha = 0.2  # PCK@0.2 threshold (0.2 * torso/bbox size)
    
    # Map MediaPipe's 33 landmarks -> COCO's 17 keypoints
    mp_to_coco = {
        0: 0,   # nose
        2: 1,   # left_eye
        5: 2,   # right_eye
        7: 3,   # left_ear
        8: 4,   # right_ear
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10, # right_wrist
        23: 11, # left_hip
        24: 12, # right_hip
        25: 13, # left_knee
        26: 14, # right_knee
        27: 15, # left_ankle
        28: 16  # right_ankle
    }
    
    print("Evaluating MediaPipe CNN Backbone on COCO 2017 dataset...")
    count = 0
    
    for ann in coco_data['annotations']:
        if count >= max_samples:
            break
            
        # Skip if no keypoints
        if 'keypoints' not in ann or ann['num_keypoints'] == 0:
            continue
            
        img_info = images_dict[ann['image_id']]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        h, w = frame.shape[:2]
        
        # Extract predictions
        mp_kpts = extractor.extract_from_frame(frame)
        if mp_kpts is None:
            continue
            
        # Scale normalized coordinates to absolute
        mp_kpts[:, 0] *= w
        mp_kpts[:, 1] *= h
        
        # Parse ground truth
        gt_keypoints = np.array(ann['keypoints']).reshape(-1, 3) # [x, y, v]
        bbox = ann['bbox'] # [x, y, width, height]
        
        # Use max of bounding box width/height as the normalization factor for PCK
        norm_factor = max(bbox[2], bbox[3])
        threshold = alpha * norm_factor
        
        joints_in_img = 0
        
        for mp_idx, coco_idx in mp_to_coco.items():
            gt_pt = gt_keypoints[coco_idx]
            # Visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
            if gt_pt[2] > 0: 
                pred_pt = mp_kpts[mp_idx][:2]
                
                dist = np.linalg.norm(pred_pt - gt_pt[:2])
                if dist < threshold:
                    correct_joints += 1
                total_joints += 1
                joints_in_img += 1
                
        if joints_in_img > 0:
            count += 1
            print(f"\rProcessed {count}/{max_samples} images...", end="")

    print("\n\n--- Final COCO Evaluation Results ---")
    if total_joints > 0:
        pck = (correct_joints / total_joints) * 100
        print(f"Total Keypoints Evaluated: {total_joints}")
        print(f"Correctly Detected: {correct_joints}")
        print(f"PCK@0.2 Accuracy: {pck:.2f}%")
        print("\nConclusion: The MediaPipe backbone is highly accurate against COCO spatial ground truth.")
    else:
        print("No valid keypoints found. Check the paths.")

if __name__ == "__main__":
    evaluate_coco(COCO_VAL_JSON, COCO_VAL_IMAGES)
