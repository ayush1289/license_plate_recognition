import cv2
import numpy as np
import pytesseract

# Load the image
img = cv2.imread("images_lp/img_1.jpg")

# Convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get the binary mask
msk = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 154]))

# Extract
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dlt = cv2.dilate(msk, krn, iterations=5)
res = 255 - cv2.bitwise_and(dlt, msk)

# OCR
txt = pytesseract.image_to_string(res, config="--psm 6")
print(txt)

# Display
cv2.imshow("res", res)
cv2.waitKey(0)