import os
import cv2

path = "2/images/selfie.jpg"
print(f"{os.path.getsize(path) / 1e6:.2f} MB")

img = cv2.imread(path)
img_small = cv2.resize(img, None, fx=0.5, fy=0.5) 
cv2.imwrite("images/selfie_small.jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 90])