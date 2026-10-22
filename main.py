# live demo. draws green/red boxes on each spot + a % full counter.
#
# first time setup:
#   1) python annotate_spots.py empty.jpg   (mark each spot once)
#   2) python capture.py                    (press R on an EMPTY lot)
#   3) python main.py
#
# u can also run it on a video file instead of the webcam:
#   python main.py --video my_lot.mp4
#
# keys: R = redo the empty refs (do this if lighting changed),
#       D = toggle yolo on/off (if installed), Q = quit
import argparse
import os
import cv2
import config
from detector.spots import (load_spots, crop_spot, load_reference_crops,
                            snapshot_reference_crops, draw_spot)
from detector.features import extract_features
from detector.scorer import decide
from detector.yolo_helper import CarDetector

GREEN, RED, AMBER = (60, 200, 80), (60, 60, 230), (60, 160, 230)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None,
                    help="video file instead of the webcam")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    spots = load_spots(config.SPOTS_FILE)
    if not spots:
        raise SystemExit("No spots.json - run annotate_spots.py first.")
    refs = load_reference_crops(config.REFERENCE_SPOTS)
    if refs is None or len(refs) != len(spots):
        raise SystemExit("Reference crops missing/stale - run capture.py and press R on an empty lot.")

    detector = CarDetector(config.YOLO_WEIGHTS, conf=0.30, imgsz=640)
    use_detector = detector.available
    if use_detector:
        print("YOLOv8n detector: ON (D to toggle)")
    else:
        print("YOLOv8n detector: not installed - shadow cues only")

    cam = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cam.isOpened():
        raise SystemExit("Could not open the webcam." if args.video is None
                         else f"Could not open video file: {args.video}")
    is_file = args.video is not None

    # only check every 8th frame, cpu cant do every frame and dont need to
    process_every = 8
    i, last_results = 0, {}
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        if i % process_every == 0:
            for k, spot in enumerate(spots):
                feats = extract_features(crop_spot(frame, spot), refs[k], config)
                last_results[k] = decide(k, feats, config,
                                         detector if use_detector else None,
                                         frame, spot)
        occupied = 0
        for k, spot in enumerate(spots):
            r = last_results.get(k)
            if r is None:
                continue
            occupied += int(r["occupied"])
            color = RED if r["occupied"] else GREEN
            label = f"#{k} {r['score']:.2f} [{r['source']}]"
            draw_spot(frame, spot, color, label)
        pct = 100.0 * occupied / max(len(spots), 1)
        cv2.putText(frame, f"Full: {pct:.0f}%   ({occupied}/{len(spots)} occupied)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER, 2, cv2.LINE_AA)
        cv2.imshow("Shadow-Aware Parking Detector", frame)

        key = cv2.waitKey(1 if not is_file else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            snapshot_reference_crops(frame, spots, config.REFERENCE_SPOTS)
            refs = load_reference_crops(config.REFERENCE_SPOTS)
            print("Re-captured empty references.")
        elif key == ord("d") and detector.available:
            use_detector = not use_detector
            print("Detector-first policy:", "ON" if use_detector else "OFF")
        i += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
