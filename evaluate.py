# no-training baseline: scores each dataset patch with classical cv only
# and checks how often it matches the real label. this is the "before"
# number that a trained model should beat later.
#
# usage:
#   python data/download.py --small
#   python evaluate.py --data data/raw/CNRPark-Patches-150x150
import os
import sys
import glob
import argparse
from collections import defaultdict

import numpy as np
import cv2


def patch_score(patch_bgr):
    # higher score = more likely a car is in the patch. no training,
    # just image stats: cars are smooth + low saturation, asphalt is speckly
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = float(lap.var())                      # overall texture
    local_std = cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2
    flat_frac = float((local_std < 40).mean())  # how much is smooth (car body)
    sat = float(hsv[:, :, 1].mean())            # asphalt is not colorful
    dark = float((gray < 60).mean())            # shadows / dark paint

    score = (0.40 * np.clip(flat_frac / 0.60, 0, 1)
             + 0.30 * np.clip(1.0 - sat / 90.0, 0, 1)
             + 0.30 * np.clip(1.0 - tex / 900.0, 0, 1))
    # a shadowed empty patch can look dark, so soften it a bit
    score -= 0.10 * np.clip(dark / 0.8, 0, 1) * np.clip(tex / 900.0, 0, 1)
    return float(score)


def iter_patches(root, limit_per_class=0):
    # walks the dataset folder and gives back (path, label, weather).
    # works for both datasets:
    #   CNR-EXT: root/PATCHES/<WEATHER>/<DATE>/cameraN/*.jpg (labels from all.txt)
    #   CNRPark: root/<CAMERA>/free|busy/*.jpg (label is the folder name)
    counts = defaultdict(int)
    patterns = [
        os.path.join(root, "PATCHES", "*", "*", "camera*", "*.jpg"),  # CNR-EXT
        os.path.join(root, "*", "free", "*.jpg"),                     # CNRPark
        os.path.join(root, "*", "busy", "*.jpg"),                     # CNRPark
    ]
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    label_map = load_label_map(root)
    unlabeled = 0
    for path in files:
        name = os.path.basename(path)
        parts = path.replace("\\", "/").split("/")
        label = None
        weather = "MIXED"
        for p in parts:
            if p in ("free", "busy"):
                label = 0 if p == "free" else 1
            if p in ("SUNNY", "OVERCAST", "RAINY"):
                weather = p
        if label is None:
            # use the official label file for CNR-EXT
            label = label_map.get(os.path.splitext(name)[0])
            if label is None:
                unlabeled += 1
                continue
        if limit_per_class and counts[(label, weather)] >= limit_per_class:
            continue
        counts[(label, weather)] += 1
        yield path, label, weather
    if unlabeled:
        print(f"[warn] {unlabeled} patches skipped: no ground-truth label found")


def load_label_map(root):
    # reads the official labels (splits.zip -> splits/CNRPark-EXT/all.txt).
    # lines look like: RAINY/2016-02-12/camera7/R_...jpg 1
    # important: dont try to guess CNR-EXT labels from the filename slot id.
    # i checked, it matches the real labels only 49.9% of the time (random)
    candidates = [
        os.path.join(root, "splits", "CNRPark-EXT", "all.txt"),
        os.path.join(os.path.dirname(os.path.abspath(root)), "splits", "CNRPark-EXT", "all.txt"),
    ]
    labels_file = next((c for c in candidates if os.path.isfile(c)), None)
    if labels_file is None:
        return {}
    label_map = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue
            rel, lab = line.rsplit(" ", 1)
            base = os.path.splitext(os.path.basename(rel))[0]
            label_map[base] = int(lab)
    print(f"ground truth: {labels_file} ({len(label_map)} labels)")
    return label_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="extracted dataset folder")
    ap.add_argument("--limit", type=int, default=0, help="max patches per class per weather")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    if not os.path.isdir(args.data):
        raise SystemExit(
            f"data root not found: {args.data}\n"
            "download the dataset first:\n"
            "  python data/download.py\n"
            "  python evaluate.py --data data/raw/CNR-EXT-Patches-150x150 --limit 300"
        )

    # weather -> [correct, total]
    stats = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for path, label, weather in iter_patches(args.data, args.limit):
        img = cv2.imread(path)
        if img is None:
            continue
        pred = 1 if patch_score(img) >= args.thresh else 0
        stats[weather][0] += int(pred == label)
        stats[weather][1] += 1
        overall[0] += int(pred == label)
        overall[1] += 1

    print(f"{'weather':<10} {'accuracy':>9} {'n':>7}")
    for weather in sorted(stats):
        c, t = stats[weather]
        if t:
            print(f"{weather:<10} {c / t:>8.1%} {t:>7}")
    if overall[1]:
        print(f"{'OVERALL':<10} {overall[0] / overall[1]:>8.1%} {overall[1]:>7}")


if __name__ == "__main__":
    main()
