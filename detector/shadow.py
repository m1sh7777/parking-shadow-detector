# small image helpers for the shadow and texture cues
import cv2
import numpy as np


def gray_blur(img, ksize=5):
    # gray + blur, basically every step needs this first
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (ksize, ksize), 0)


def texture_energy(patch_gray):
    # laplacian variance. speckly asphalt = high number, smooth car = low
    lap = cv2.Laplacian(patch_gray, cv2.CV_64F)
    return float(lap.var())


def shadow_fraction(cur_bgr, ref_bgr=None, v_ratio=0.60, s_max=140):
    # fraction of the patch that is dark but not colorful (shadow looking).
    # if u give it a ref patch it looks for pixels darker than the ref,
    # otherwise it just looks for dark pixels. returns (fraction, mask)
    hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    if ref_bgr is not None:
        # pixel counts as shadow if its much darker than the ref was
        hsv_r = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
        v_ref = hsv_r[:, :, 2].astype(np.float32)
        mask = (v < v_ratio * np.maximum(v_ref, 1.0)) & (s < s_max)
    else:
        mask = (v < 80) & (s < s_max)
    frac = float(mask.mean())
    return frac, mask.astype(np.uint8) * 255


def changed_fraction(cur_gray, ref_gray, thresh=30):
    # fraction of pixels that changed more than thresh vs the ref
    diff = cv2.absdiff(cur_gray, ref_gray)
    _, m = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return float(cv2.countNonZero(m)) / float(m.size), diff
