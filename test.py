import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow("Test Window", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()