# importing modules
import cv2
import numpy as np

# cascade classifier used to detect rectangle or license plate
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500


count = "12"
# cv2.imread is used to read the image file.
img = cv2.imread(f"dataset_lp/{count}.jpeg")

name = f"img_{count}"

while True:

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y-5),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgRoi)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("images_lp/"+f"{name}"+".jpg", imgRoi)
        cv2.imshow("Result", img)
        # cv2.waitKey(500)
        break
cv2.destroyAllWindows()
# count+=1
