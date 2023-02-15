import cv2

cap = cv2.imread("dataset_lp/1.jpeg")
# cv2.imshow('img',cap)

while True:
    # _, img = cap.read()
    
    cv2.imshow("Image", cap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()