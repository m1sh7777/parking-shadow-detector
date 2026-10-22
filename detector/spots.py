# load / save the spot boxes and the empty reference crops
import numpy as np
import cv2


def load_spots(path):
    # reads spots.json, returns [] if its not there yet
    import json, os
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def crop_spot(frame, spot):
    # cut the spot box out of the frame (clamped so it cant go out of bounds)
    x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]
    H, W = frame.shape[:2]
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return frame[y:y + h, x:x + w]


def snapshot_reference_crops(frame, spots, out_path):
    # save a crop of each spot from the CURRENT (empty) frame. this is the
    # reference the detector compares against later
    crops = np.stack([crop_spot(frame, s) for s in spots]).astype(np.uint8)
    np.savez_compressed(out_path, crops=crops)
    return out_path


def load_reference_crops(path):
    # returns the saved empty crops, or None if u havent made them yet
    import os
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["crops"]


def draw_spot(frame, spot, color, label=""):
    # draws the box + label on the frame (green = free, red = car)
    x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(frame, label, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame
