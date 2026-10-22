# optional yolo wrapper. u need to pip install ultralytics for this to work.
# if u dont have it everything returns None and the program just uses the
# shadow cues instead, so this is never a hard requirement
import numpy as np

# these are the vehicle classes in the coco dataset yolo was trained on
CLASS_COCO_VEHICLES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}


class CarDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.30, imgsz=640):
        try:
            from ultralytics import YOLO   # lazy import so it stays optional
        except ImportError:
            self.model = None
            return
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    @property
    def available(self):
        return self.model is not None

    def detect(self, frame_bgr):
        # returns a list of (x, y, w, h, conf, class) for vehicles in the frame
        if self.model is None:
            return []
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False, device="cpu")
        boxes = []
        for r in res:
            for b in r.boxes:
                cls = int(b.cls.item())
                if cls in CLASS_COCO_VEHICLES:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    boxes.append((x1, y1, x2 - x1, y2 - y1,
                                  float(b.conf.item()), cls))
        return boxes

    def spot_confidence(self, frame_bgr, spot):
        # best vehicle confidence for this spot. None if no model
        if self.model is None:
            return None
        x, y, w, h = spot["x"], spot["y"], spot["w"], spot["h"]
        best = None
        for bx, by, bw, bh, conf, _cls in self.detect(frame_bgr):
            # how much of the detection overlaps the spot box
            ix = max(0, min(x + w, bx + bw) - max(x, bx))
            iy = max(0, min(y + h, by + bh) - max(y, by))
            inter = ix * iy
            if inter <= 0:
                continue
            # if the car center is inside the spot thats the strong signal
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if x <= cx <= x + w and y <= cy <= y + h:
                c = conf + 0.05 * (inter / float(w * h))  # small overlap bonus
            else:
                c = 0.5 * conf * (inter / float(w * h))
            best = c if best is None else max(best, c)
        return best
