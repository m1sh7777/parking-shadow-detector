# smoke tests. no camera, no dataset, no yolo needed, all fake images.
#   pip install pytest httpx
#   pytest tests/ -v
import json
import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detector.shadow import texture_energy, shadow_fraction, changed_fraction
from detector.scorer import cue_score, decide


# fake patches for testing

def noisy_asphalt(seed=0, size=(64, 64)):
    # speckled gray = stands in for empty asphalt
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(120, 25, size), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def smooth_car_patch(size=(64, 64)):
    # flat dark patch = stands in for a car covering the asphalt
    rng = np.random.default_rng(1)
    img = np.clip(np.full(size, 45.0) + rng.normal(0, 2, size), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# cue tests

def test_texture_energy_separates_asphalt_from_car():
    asphalt = texture_energy(cv2.cvtColor(noisy_asphalt(), cv2.COLOR_BGR2GRAY))
    car = texture_energy(cv2.cvtColor(smooth_car_patch(), cv2.COLOR_BGR2GRAY))
    assert asphalt > 5 * car, f"asphalt tex={asphalt:.1f} should dominate car tex={car:.1f}"


def test_shadow_fraction_higher_for_dark_unsaturated_patch():
    _, mask_free = shadow_fraction(noisy_asphalt())
    _, mask_car = shadow_fraction(smooth_car_patch())
    assert mask_car.mean() > mask_free.mean()


def test_changed_fraction_zero_for_identical_patches():
    g = cv2.cvtColor(noisy_asphalt(), cv2.COLOR_BGR2GRAY)
    frac, _ = changed_fraction(g, g, 30)
    assert frac == 0.0


def test_changed_fraction_detects_parked_car():
    cur = cv2.cvtColor(smooth_car_patch(), cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(noisy_asphalt(), cv2.COLOR_BGR2GRAY)
    frac, _ = changed_fraction(cur, ref, 30)
    assert frac > 0.5


# scoring + policy tests

def _feats(**kw):
    base = dict(diff_frac=0.0, texture_loss=0.0, shadow_now=0.0,
                shadow_delta=0.0, brightness_ratio=1.0)
    base.update(kw)
    return base


def test_free_features_score_below_threshold():
    assert cue_score(_feats(), config) < config.OCCUPANCY_THRESHOLD


def test_car_features_score_above_threshold():
    feats = _feats(diff_frac=0.5, texture_loss=0.9, shadow_delta=0.3)
    assert cue_score(feats, config) >= config.OCCUPANCY_THRESHOLD


def test_cloud_guard_damps_global_darkness():
    # everything went dark BUT texture is still there = cloud, not a car
    cloudy = _feats(diff_frac=0.6, texture_loss=0.1, shadow_delta=0.3,
                    brightness_ratio=0.5)
    parked = _feats(diff_frac=0.6, texture_loss=0.9, shadow_delta=0.3,
                    brightness_ratio=0.5)
    assert cue_score(cloudy, config) < cue_score(parked, config)


def test_decide_without_detector_reports_cue_source():
    out = decide(0, _feats(diff_frac=0.5, texture_loss=0.9), config)
    assert out["source"] == "shadow_cues"
    assert isinstance(out["occupied"], bool)
    assert "features" in out


# api tests (no server needed, fastapi test client)

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from api.main import app
    spots = [{"x": 10, "y": 10, "w": 64, "h": 64},
             {"x": 120, "y": 10, "w": 64, "h": 64}]
    (tmp_path / "spots.json").write_text(json.dumps(spots))
    monkeypatch.setattr(config, "SPOTS_FILE", str(tmp_path / "spots.json"))
    monkeypatch.setattr(config, "REFERENCE_SPOTS", str(tmp_path / "ref.npz"))
    return TestClient(app)


def _jpg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_reports_spots_and_reference_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["spots"] == 2
    assert body["reference_loaded"] is False


def test_reference_then_analyze_round_trip(client):
    empty = noisy_asphalt(seed=7, size=(200, 200))   # fake empty lot
    r = client.post("/reference", files={"frame": ("f.jpg", _jpg(empty))})
    assert r.status_code == 200
    assert r.json()["spots"] == 2

    # same frame again = everything should be free
    body = client.post("/analyze", files={"frame": ("f.jpg", _jpg(empty))}).json()
    assert body["occupancy"]["occupied"] == 0
    assert body["occupancy"]["total"] == 2

    # park a fake car over spot 0 only
    frame = empty.copy()
    cv2.rectangle(frame, (10, 10), (74, 74), (40, 40, 45), -1)
    body = client.post("/analyze", files={"frame": ("f.jpg", _jpg(frame))}).json()
    assert body["spots"][0]["occupied"] is True
    assert body["spots"][1]["occupied"] is False
    assert body["occupancy"]["percent_full"] == 50.0


def test_analyze_before_reference_returns_error(client):
    body = client.post("/analyze", files={"frame": ("f.jpg", _jpg(noisy_asphalt()))}).json()
    assert "error" in body
