# takes the cue features and turns them into one score, then decides.
# policy: if yolo is installed and confident it wins, else my cue score decides
import numpy as np


def cue_score(feats, cfg):
    # fuse the 3 cues into one score between 0 and 1
    diff_term = float(np.clip(feats["diff_frac"] / 0.35, 0.0, 1.0))
    texture_term = float(np.clip(feats["texture_loss"], 0.0, 1.0))
    # small shadow drift happens all the time so ignore tiny changes
    shadow_term = float(np.clip((feats["shadow_delta"] - 0.05) / 0.40, 0.0, 1.0))

    score = (cfg.W_DIFF * diff_term
             + cfg.W_TEXTURE * texture_term
             + cfg.W_SHADOW * shadow_term)

    # cloud check: if the whole patch got dark but texture is still there
    # its just a cloud passing, not a car. cut the score in half for that
    if feats["brightness_ratio"] < 0.7 and feats["texture_loss"] < 0.3:
        score *= 0.5
    return score


def decide(spot_id, feats, cfg, detector=None, cur_frame=None, spot=None):
    # returns the decision for one spot as a dict.
    # if the detector says conf >= YOLO_CONF that answer wins, otherwise
    # fall back to the cue score
    score = cue_score(feats, cfg)
    cue_occupied = score >= cfg.OCCUPANCY_THRESHOLD

    out = {
        "spot": spot_id,
        "score": round(score, 3),
        "occupied": bool(cue_occupied),
        "source": "shadow_cues",
        "detector_conf": None,
        "features": {k: round(v, 4) for k, v in feats.items()},
    }

    if detector is not None and cur_frame is not None and spot is not None:
        conf = detector.spot_confidence(cur_frame, spot)
        out["detector_conf"] = None if conf is None else round(conf, 3)
        if conf is not None and conf >= cfg.YOLO_CONF:
            out["occupied"] = bool(conf >= cfg.YOLO_CONF)
            out["source"] = "yolo"
        else:
            # yolo is unsure (maybe the car is behind a tree) so my cues decide
            out["occupied"] = bool(cue_occupied)
            out["source"] = "yolo_lowconf_fallback"
    return out
