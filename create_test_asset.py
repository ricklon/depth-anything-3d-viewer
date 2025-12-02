
import cv2
import numpy as np

# Create a test pattern image
h, w = 720, 1280
img = np.zeros((h, w, 3), dtype=np.uint8)

# Grid
step = 100
img[::step, :] = [50, 50, 50]
img[:, ::step] = [50, 50, 50]

# Circles
cv2.circle(img, (w//2, h//2), 200, (0, 255, 0), 2)
cv2.circle(img, (w//2, h//2), 100, (0, 0, 255), 2)

# Text
cv2.putText(img, "DA3D Projection Test", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

cv2.imwrite("assets/test_pattern.png", img)
print("Created assets/test_pattern.png")
