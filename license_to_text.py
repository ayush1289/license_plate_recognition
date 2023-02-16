import cv2
import numpy as np
import pytesseract

# Load the image
img = cv2.imread("images_lp/img_1.jpg")
# img = cv2.resize(img,(300,150))
# Convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get the binary mask
msk = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 154]))

# Extract
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dlt = cv2.dilate(msk, krn, iterations=5,)
res = 255 - cv2.bitwise_and(dlt, msk)

# OCR
txt = pytesseract.image_to_string(res, config="--psm 7").replace(" ","")
print(txt)

# Display
cv2.imshow("res", res)
cv2.waitKey(0)


