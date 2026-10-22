# this file compares the current spot crop vs its empty reference crop
import numpy as np
from detector.shadow import gray_blur, texture_energy, shadow_fraction, changed_fraction


def extract_features(cur_patch_bgr, ref_patch_bgr, cfg):
    # every cue is "now vs this same spot when it was empty". doing it this
    # way cancels out stuff like asphalt tone and angle for each spot
    cur_g = gray_blur(cur_patch_bgr)
    ref_g = gray_blur(ref_patch_bgr)

    diff_frac, _ = changed_fraction(cur_g, ref_g, cfg.DIFF_THRESH)

    tex_cur = texture_energy(cur_g)
    tex_ref = max(texture_energy(ref_g), cfg.TEXTURE_EPS)
    # 1.0 means the asphalt texture is fully gone (car is sitting on it)
    texture_loss = float(np.clip(1.0 - tex_cur / tex_ref, 0.0, 1.0))

    shadow_now, _ = shadow_fraction(cur_patch_bgr, ref_patch_bgr,
                                    cfg.SHADOW_V_RATIO, cfg.SHADOW_S_MAX)
    shadow_ref, _ = shadow_fraction(ref_patch_bgr, None,
                                    cfg.SHADOW_V_RATIO, cfg.SHADOW_S_MAX)
    shadow_delta = abs(shadow_now - shadow_ref)

    # if the whole patch just got darker its probably a cloud not a car
    brightness_ratio = float(cur_g.mean()) / max(float(ref_g.mean()), 1.0)

    return {
        "diff_frac": diff_frac,
        "texture_loss": texture_loss,
        "shadow_now": shadow_now,
        "shadow_delta": shadow_delta,
        "brightness_ratio": brightness_ratio,
    }
