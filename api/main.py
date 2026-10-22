# this is the api version of the detector. same logic as main.py but over http.
#
# run it:
#   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
#
# endpoints:
#   GET  /health     -> is it alive + how many spots
#   GET  /spots      -> the spot boxes
#   POST /reference  -> upload an EMPTY lot pic to set the reference
#   POST /analyze    -> upload a frame, get back which spots have cars
import os
import sys
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detector.spots import load_spots, crop_spot, snapshot_reference_crops, load_reference_crops
from detector.features import extract_features
from detector.scorer import decide
from detector.yolo_helper import CarDetector

app = FastAPI(title="Shadow-Aware Parking Detector", version="0.1.0")
detector = CarDetector(config.YOLO_WEIGHTS, conf=0.30, imgsz=640)


def _decode(data: bytes):
    # bytes -> opencv image, or error if its not a real image
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (jpg/png expected).")
    return img


@app.get("/health")
def health():
    return {
        "status": "ok",
        "spots": len(load_spots(config.SPOTS_FILE)),
        "reference_loaded": load_reference_crops(config.REFERENCE_SPOTS) is not None,
        "yolo_available": detector.available,
    }


@app.get("/spots")
def spots():
    return {"spots": load_spots(config.SPOTS_FILE)}


@app.post("/reference")
def reference(frame: UploadFile = File(...)):
    img = _decode(frame.file.read())
    sp = load_spots(config.SPOTS_FILE)
    if not sp:
        return {"error": "no spots.json - run annotate_spots.py first"}
    path = snapshot_reference_crops(img, sp, config.REFERENCE_SPOTS)
    return {"saved": path, "spots": len(sp)}


@app.post("/analyze")
def analyze(frame: UploadFile = File(...)):
    img = _decode(frame.file.read())
    sp = load_spots(config.SPOTS_FILE)
    refs = load_reference_crops(config.REFERENCE_SPOTS)
    if not sp or refs is None or len(refs) != len(sp):
        return {"error": "missing spots.json or reference crops"}
    results = []
    for k, spot in enumerate(sp):
        feats = extract_features(crop_spot(img, spot), refs[k], config)
        results.append(decide(k, feats, config, detector, img, spot))
    occupied = sum(1 for r in results if r["occupied"])
    return {
        "occupancy": {"occupied": occupied, "total": len(sp),
                      "percent_full": round(100.0 * occupied / len(sp), 1)},
        "spots": results,
    }
