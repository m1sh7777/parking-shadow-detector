# grabs frames from the webcam and saves them.
#   E = save empty.jpg (empty lot, this is the reference)
#   O = save occupied.jpg (for the basic diff demo)
#   R = save the empty crops for each spot (needs spots.json first)
#   Q = quit
import os
import cv2
import config
from detector.spots import load_spots, snapshot_reference_crops


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam. Close other apps using it and retry.")

    print("E = save empty.jpg | O = save occupied.jpg | R = snapshot spot references | Q = quit")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        cv2.imshow("Webcam - E: empty, O: occupied, R: references, Q: quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("e"):
            cv2.imwrite("empty.jpg", frame)
            print("Saved empty.jpg")
        elif key == ord("o"):
            cv2.imwrite("occupied.jpg", frame)
            print("Saved occupied.jpg")
        elif key == ord("r"):
            spots = load_spots(config.SPOTS_FILE)
            if not spots:
                print("No spots.json found - run annotate_spots.py first.")
            else:
                path = snapshot_reference_crops(frame, spots, config.REFERENCE_SPOTS)
                print("Saved per-spot reference crops to", path)
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
