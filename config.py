# all the tunable numbers live here. change these if ur setup is different
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# files the program makes / reads
SPOTS_FILE = os.path.join(BASE_DIR, "spots.json")          # spot boxes u draw
REFERENCE_FRAME = os.path.join(BASE_DIR, "reference.jpg")  # empty lot pic
REFERENCE_SPOTS = os.path.join(BASE_DIR, "reference_spots.npz")  # saved empty crops per spot

# dataset stuff (this is the no-training part of the project)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
# the dataset moved off cnrpark.it, the old links are dead. this github release
# link is the official one now (took me a while to find this lol)
MIRROR = "https://github.com/fabiocarrara/deep-parking/releases/download/archive"
CNR_EXT_URL = f"{MIRROR}/CNR-EXT-Patches-150x150.zip"   # 449.5 MB
CNR_URL = f"{MIRROR}/CNRPark-Patches-150x150.zip"       # 36.6 MB
SPLITS_URL = f"{MIRROR}/splits.zip"                     # 27.2 MB (optional)

# thresholds i picked by testing. not perfect but they work
DIFF_THRESH = 30          # how much a pixel must change to count as changed
SHADOW_V_RATIO = 0.60     # pixel is shadow if its way darker than the ref
SHADOW_S_MAX = 140        # shadows are not colorful so skip really saturated px
TEXTURE_EPS = 1e-4        # tiny floor so we never divide by 0

# how much each cue counts for the final score (adds up to 1)
W_TEXTURE = 0.45          # asphalt speckle disappears when a car sits on it
W_DIFF = 0.35             # pixels that changed vs the empty ref
W_SHADOW = 0.20           # shadow pattern changed

OCCUPANCY_THRESHOLD = 0.50  # score above this = car is there

# optional yolo. only used if u do pip install ultralytics
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")
YOLO_CONF = 0.45          # above this trust yolo, below it use my cue score
