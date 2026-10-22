# draw boxes around parking spots with the mouse, saves to spots.json.
# enter/space = accept the box, esc = redo it, close window or q = done.
#
# usage:
#   python annotate_spots.py              (grabs a frame from the webcam)
#   python annotate_spots.py empty.jpg    (or use a photo)
import os
import sys
import json
import cv2


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise SystemExit(f"Could not read image {sys.argv[1]}")
    else:
        cam = cv2.VideoCapture(0)
        ok, img = cam.read()
        cam.release()
        if not ok:
            raise SystemExit("Could not grab a frame from the webcam.")

    spots = []
    print("Draw a box per spot -> ENTER/SPACE to accept, ESC to redo. Q to finish.")
    while True:
        r = cv2.selectROI("Draw parking spots (Q to finish)", img, showCrosshair=False)
        cv2.destroyAllWindows()
        if r == (0, 0, 0, 0):   # means u pressed q or closed the window
            break
        x, y, w, h = (int(v) for v in r)
        if w and h:
            spots.append({"id": len(spots), "x": x, "y": y, "w": w, "h": h})
            print(f"  spot {len(spots) - 1}: x={x} y={y} w={w} h={h}")

    with open("spots.json", "w") as f:
        json.dump(spots, f, indent=2)
    print(f"Saved {len(spots)} spots to spots.json")


if __name__ == "__main__":
    main()
