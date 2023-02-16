import cv2
import numpy as np
import pytesseract

# Load the image 1,4,5,6,8,12
img = cv2.imread("images_lp/img_12.jpg")
img = cv2.resize(img,(170,60))
# Convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Get the binary mask
msk = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([190, 255, 154]))

# Extract
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dlt = cv2.dilate(msk, krn, iterations=5,)
res = 255 - cv2.bitwise_and(dlt, msk)

# OCR
txt = pytesseract.image_to_string(res, config="--psm 7").replace(" ","")
print(txt)

# Display
cv2.imshow("res", res)
cv2.waitKey(0)


