import os
import cv2
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and retry.")
 
print("Press E = save empty.jpg | O = save occupied.jpg | Q = quit")
 
while True:
    ok, frame = cam.read()
    if not ok:
        break
    cv2.imshow("Webcam - E: empty, O: occupied, Q: quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("e"):
        cv2.imwrite("empty.jpg", frame)
        print("Saved empty.jpg")
    elif key == ord("o"):
        cv2.imwrite("occupied.jpg", frame)
        print("Saved occupied.jpg")
    elif key == ord("q"):
        break
 
cam.release()
cv2.destroyAllWindows()