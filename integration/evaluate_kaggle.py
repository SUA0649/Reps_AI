import numpy as np
import pandas as pd
import cv2
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from cnn_module.keypoint_extractor import KeypointExtractor

# ==========================================
# USER: UPDATE THESE PATHS!
KAGGLE_CSV_PATH = "REPLACE_WITH_PATH_TO_KAGGLE_ANNOTATIONS.csv"
KAGGLE_IMAGES_DIR = "REPLACE_WITH_PATH_TO_KAGGLE_IMAGES"
# ==========================================

def evaluate_kaggle_csv(csv_path, images_dir, max_samples=500):
    """
    Evaluates MediaPipe against a standard Kaggle Pose CSV file.
    Assumes the CSV has an 'image' or 'filename' column, and columns for keypoint coordinates.
    Adjust the parsing logic below based on your specific CSV structure.
    """
    print(f"Loading Kaggle annotations from {csv_path}...")
    if not os.path.exists(csv_path):
        print("❌ Error: Update the placeholders in the script with the actual paths!")
        return

    df = pd.read_csv(csv_path)
    extractor = KeypointExtractor()
    
    total_joints = 0
    correct_joints = 0
    
    # We will use Mean Absolute Error (MAE) for normalized coordinates,
    # or PCK if the CSV provides absolute pixel coordinates.
    # We'll calculate the Average Distance Error in pixels.
    total_pixel_error = 0.0
    
    print("Evaluating MediaPipe CNN Backbone on Kaggle CSV dataset...")
    count = 0
    
    for idx, row in df.iterrows():
        if count >= max_samples:
            break
            
        # Try to find the image filename column
        # Standard names: 'image', 'filename', 'image_path', 'file'
        img_col = next((col for col in ['image', 'filename', 'image_path', 'file', 'id'] if col in row.keys()), None)
        if not img_col:
            print("Could not find image filename column in CSV. Please update the script.")
            break
            
        img_name = str(row[img_col])
        # Ensure it has an extension
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.jpg'
            
        img_path = os.path.join(images_dir, img_name)
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
            
        # Parse ground truth from CSV row
        # This part heavily depends on your specific Kaggle CSV.
        # Assuming the CSV has columns like 'x0', 'y0', 'x1', 'y1' ... 
        # representing the 33 MediaPipe/COCO points.
        
        frame_error = 0.0
        joints_in_img = 0
        
        for i in range(33):
            # Many datasets use 'x{id}' and 'y{id}' or 'point_{id}_x'
            x_col = f"x{i}" if f"x{i}" in row.keys() else f"point_{i}_x" if f"point_{i}_x" in row.keys() else None
            y_col = f"y{i}" if f"y{i}" in row.keys() else f"point_{i}_y" if f"point_{i}_y" in row.keys() else None
            
            if x_col and y_col and not pd.isna(row[x_col]) and not pd.isna(row[y_col]):
                # Ground truth (assuming normalized 0-1 based on typical Kaggle MediaPipe datasets)
                gt_x = float(row[x_col]) * w if float(row[x_col]) <= 1.0 else float(row[x_col])
                gt_y = float(row[y_col]) * h if float(row[y_col]) <= 1.0 else float(row[y_col])
                
                # Prediction
                pred_x = mp_kpts[i, 0] * w
                pred_y = mp_kpts[i, 1] * h
                
                dist = np.linalg.norm([pred_x - gt_x, pred_y - gt_y])
                frame_error += dist
                total_joints += 1
                joints_in_img += 1
                
                # If distance is less than 5% of the image width, count it as "correct" (PCK@0.05)
                if dist < (0.05 * w):
                    correct_joints += 1
                    
        if joints_in_img > 0:
            total_pixel_error += (frame_error / joints_in_img)
            count += 1
            print(f"\rProcessed {count}/{max_samples} images...", end="")

    print("\n\n--- Final Kaggle CSV Evaluation Results ---")
    if count > 0 and total_joints > 0:
        pck = (correct_joints / total_joints) * 100
        avg_pixel_error = total_pixel_error / count
        print(f"Total Images Evaluated: {count}")
        print(f"Total Keypoints Evaluated: {total_joints}")
        print(f"Average Pixel Error per Joint: {avg_pixel_error:.2f} pixels")
        print(f"PCK@0.05 (Tolerance of 5% Image Width): {pck:.2f}%")
        print("\nNote: Accuracy depends highly on whether the Kaggle dataset ground truth matches MediaPipe's exact keypoint topology.")
    else:
        print("No valid keypoints found. Check the CSV column names and parsing logic.")

if __name__ == "__main__":
    evaluate_kaggle_csv(KAGGLE_CSV_PATH, KAGGLE_IMAGES_DIR)
