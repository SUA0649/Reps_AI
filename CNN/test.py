import numpy as np
import cv2
import torch  # or: import torch

print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("Torch version:", torch.__version__)  # or: torch.__version__

# Test webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("✅ Webcam working!")
    cv2.imshow('Test', frame)
    cv2.waitKey(1000)
else:
    print("❌ Webcam not detected")
cap.release()
cv2.destroyAllWindows()
