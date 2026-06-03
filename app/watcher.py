import os
import time
import uuid
import shutil
import logging
import subprocess
import glob
import requests
import cv2
import numpy as np
from urllib.parse import quote

from database import insert_event
from analyzer import extract_dominant_color_from_frame
from lpr import PlateAnalyzer

log = logging.getLogger("watcher")

CAMERA_URL = os.environ.get("CAMERA_URL", "http://192.168.1.44")
CAMERA_USER = os.environ.get("CAMERA_USER", "admin")
CAMERA_PASS = os.environ.get("CAMERA_PASS", "")
CAMERA_RTSP = os.environ.get("CAMERA_RTSP", "")
CAMERA_NAME = os.environ.get("CAMERA_NAME", "portail")
SNAPSHOTS_DIR = os.environ.get("SNAPSHOTS_DIR", "/data/snapshots")
EVENTS_DIR = os.environ.get("EVENTS_DIR", "/data/events")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))
CAPTURE_DURATION = int(os.environ.get("CAPTURE_DURATION", "15"))
CAPTURE_FPS = int(os.environ.get("CAPTURE_FPS", "5"))
COOLDOWN = int(os.environ.get("COOLDOWN", "60"))
PLATERECOGNIZER_KEY = os.environ.get("PLATERECOGNIZER_API_KEY", "")

_token: str | None = None
_token_time = 0.0
_last_event_time = 0.0
_analyzer: PlateAnalyzer | None = None

import re as _re
import json as _json

_FR_PLATE_RE = _re.compile(r'^([A-Z]{2})(\d{3})([A-Z]{2})$')
_ZONE_PATH = "/data/zone.json"


def _read_zone_points() -> list | None:
    try:
        with open(_ZONE_PATH) as f:
            pts = _json.load(f).get("points", [])
        if len(pts) >= 3:
            return pts
    except Exception:
        pass
    return None


def _zone_mask(h: int, w: int, pts: list) -> np.ndarray:
    poly = np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def _normalize_plate(raw: str) -> str:
    """Validate and format a French plate (6-8 alphanumeric chars).
    Standard new format AB-123-CD is detected and hyphenated automatically."""
    alnum = "".join(c for c in raw.upper() if c.isalnum())
    if len(alnum) < 6 or len(alnum) > 8:
        return ""
    m = _FR_PLATE_RE.match(alnum)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return alnum


def _rtsp_url() -> str:
    if CAMERA_RTSP:
        return CAMERA_RTSP
    host = CAMERA_URL.replace("http://", "").replace("https://", "").split(":")[0]
    return f"rtsp://{quote(CAMERA_USER, safe='')}:{quote(CAMERA_PASS, safe='')}@{host}/h264Preview_01_main"


def _login() -> str | None:
    global _token, _token_time
    if _token and (time.time() - _token_time) < 3000:
        return _token
    try:
        resp = requests.post(
            f"{CAMERA_URL}/api.cgi?cmd=Login",
            json=[{"cmd": "Login", "param": {"User": {"userName": CAMERA_USER, "password": CAMERA_PASS}}}],
            timeout=5,
        )
        data = resp.json()
        if data[0]["code"] == 0:
            _token = data[0]["value"]["Token"]["name"]
            _token_time = time.time()
            return _token
    except Exception as e:
        log.warning(f"Login error: {e}")
    return None


def _get_ai_state() -> dict | None:
    token = _login()
    if not token:
        return None
    try:
        resp = requests.post(
            f"{CAMERA_URL}/api.cgi?cmd=GetAiState&token={token}",
            json=[{"cmd": "GetAiState", "action": 0, "param": {"channel": 0}}],
            timeout=5,
        )
        data = resp.json()
        if data[0]["code"] == 0:
            return data[0]["value"]
    except Exception as e:
        log.warning(f"GetAiState error: {e}")
    return None


def _vehicle_crop(frame: np.ndarray, plate_bbox: tuple | None) -> np.ndarray:
    """Crop around the vehicle using the plate bbox as anchor.
    Expands generously above/around the plate where the vehicle body is."""
    h, w = frame.shape[:2]
    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        pw, ph = px2 - px1, py2 - py1
        pad_x  = max(int(pw * 3.5), 300)
        pad_up = max(int(ph * 9),   400)  # vehicle extends above the plate
        pad_dn = max(int(ph * 2.5), 100)
        cx1 = max(0, px1 - pad_x)
        cx2 = min(w, px2 + pad_x)
        cy1 = max(0, py1 - pad_up)
        cy2 = min(h, py2 + pad_dn)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.shape[0] >= 80 and crop.shape[1] >= 80:
            return crop
    # Fallback: center 60% of frame
    cy1, cy2 = int(h * 0.1), int(h * 0.9)
    cx1, cx2 = int(w * 0.1), int(w * 0.9)
    return frame[cy1:cy2, cx1:cx2]


def _vehicle_color(frame: np.ndarray, plate_bbox: tuple | None) -> tuple[str, str]:
    """Extract dominant color from the vehicle body (above the plate, or center frame)."""
    h, w = frame.shape[:2]
    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        pw = px2 - px1
        ph = py2 - py1
        crop_x1 = max(0, px1 - pw * 2)
        crop_x2 = min(w, px2 + pw * 2)
        crop_y2 = max(0, py1 - 5)
        crop_y1 = max(0, py1 - ph * 10)
        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
            crop = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
            return extract_dominant_color_from_frame(crop)
    cy1 = h // 4
    cy2 = 3 * h // 4
    cx1 = w // 4
    cx2 = 3 * w // 4
    return extract_dominant_color_from_frame(frame[cy1:cy2, cx1:cx2])


def _frame_score(frame: np.ndarray, plates: list) -> float:
    if plates:
        x1, y1, x2, y2, conf = plates[0]
        return (x2 - x1) * (y2 - y1) * conf
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() * 0.001


def _process_clip(event_id: str, event_dir: str, clip_path: str, camera_name: str = None):
    """Process an existing clip: extract frames at full res, transcode for browser, LPR, store in DB."""
    if camera_name is None:
        camera_name = CAMERA_NAME

    frames_pattern = os.path.join(event_dir, "frame_%04d.jpg")

    # Extract frames at full original resolution BEFORE any transcode
    subprocess.run([
        "ffmpeg", "-y", "-i", clip_path,
        "-vf", f"fps={CAPTURE_FPS}", "-q:v", "2", frames_pattern,
    ], capture_output=True, timeout=60)

    # Transcode to H.264 baseline 720p for browser (after frame extraction)
    web_clip = os.path.join(event_dir, "clip_web.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", clip_path,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.1",
        "-preset", "fast", "-crf", "28",
        "-vf", "scale=-2:720",
        "-an",
        "-movflags", "+faststart",
        web_clip,
    ], capture_output=True, timeout=120)
    if os.path.exists(web_clip):
        os.replace(web_clip, clip_path)
    else:
        log.warning("Transcode failed, keeping raw clip")

    frame_files = sorted(glob.glob(os.path.join(event_dir, "frame_*.jpg")))
    log.info(f"Extracted {len(frame_files)} frames")

    if not frame_files:
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    # Pass 1: load all frames, compute sharpness + inter-frame motion
    # Motion detection uses a downscaled gray image (fast) to find frames where
    # the vehicle is present — a moving car causes high frame diff even when blurry.
    _MOTION_W = 320
    loaded: list[tuple[np.ndarray, str, float, float]] = []  # frame, path, sharpness, motion
    prev_small: np.ndarray | None = None

    zone_pts = _read_zone_points()
    zone_mask_small: np.ndarray | None = None  # computed lazily on first frame

    for path in frame_files:
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        h, w = gray.shape
        small = cv2.resize(gray, (_MOTION_W, _MOTION_W * h // w))
        if zone_mask_small is None and zone_pts:
            sh, sw = small.shape
            zone_mask_small = cv2.resize(_zone_mask(h, w, zone_pts), (sw, sh))
        if prev_small is not None:
            diff = np.abs(small.astype(np.float32) - prev_small.astype(np.float32))
            if zone_mask_small is not None:
                diff = diff * (zone_mask_small.astype(np.float32) / 255.0)
            motion = float(np.mean(diff))
        else:
            motion = 0.0
        prev_small = small
        loaded.append((frame, path, lap, motion))

    # Frame 0 has no predecessor — inherit frame 1's motion so car entering on first frame isn't missed
    if len(loaded) > 1:
        loaded[0] = (loaded[0][0], loaded[0][1], loaded[0][2], loaded[1][3])

    max_sharp = max((x[2] for x in loaded), default=1.0) or 1.0
    max_motion = max((x[3] for x in loaded), default=1.0) or 1.0

    # Combined score: motion weighted higher to prioritise car-present frames,
    # sharpness as tiebreaker within those frames.
    combined: list[tuple[float, np.ndarray, str]] = sorted(
        [(0.65 * m / max_motion + 0.35 * s / max_sharp, frame, path)
         for frame, path, s, m in loaded],
        key=lambda x: -x[0],
    )

    # Pass 2: YOLO on top 15 by combined score → plate-area × conf ranking
    top15 = combined[:15]
    scored: list[tuple[float, np.ndarray, str]] = []
    for _, frame, path in top15:
        plates = _analyzer.detect_plates(frame) if _analyzer else []
        if plates and zone_pts:
            fh, fw = frame.shape[:2]
            poly = np.array([[int(x * fw), int(y * fh)] for x, y in zone_pts], dtype=np.int32)
            plates = [p for p in plates
                      if cv2.pointPolygonTest(poly.reshape(-1, 1, 2),
                                              ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2),
                                              False) >= 0]
        score = _frame_score(frame, plates)
        scored.append((score, frame, path))
    scored.sort(key=lambda x: -x[0])

    # Merge: YOLO-scored top15 first, then remaining frames by combined score
    scored_paths = {path for _, _, path in scored}
    rest = [(score, frame, path) for score, frame, path in combined if path not in scored_paths]
    full_sorted = scored + rest

    for i, (_, _, old_path) in enumerate(full_sorted):
        os.rename(old_path, old_path + ".tmp")
    for i, (_, _, old_path) in enumerate(full_sorted):
        os.rename(old_path + ".tmp", os.path.join(event_dir, f"frame_{i+1:04d}.jpg"))

    best_frame = full_sorted[0][1] if full_sorted else None
    if best_frame is None:
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    # LPR: PlateRecognizer API (top 3 frames) → fallback to local
    plate, conf, plate_bbox, plate_corrected = ("", 0.0, None, None)
    if PLATERECOGNIZER_KEY:
        from lpr import call_platerecognizer
        for _, frame, _ in scored[:3]:
            p, c = call_platerecognizer(frame, PLATERECOGNIZER_KEY)
            p = _normalize_plate(p)
            if p and c > conf:
                plate, conf = p, c
            if conf >= 0.7:
                break
        if plate:
            # Get local perspective-corrected crop for display
            if _analyzer:
                plates = _analyzer.detect_plates(best_frame)
                if plates:
                    x1, y1, x2, y2, _ = plates[0]
                    quad = _analyzer._find_plate_quad(best_frame, int(x1), int(y1), int(x2), int(y2))
                    plate_bbox = (int(x1), int(y1), int(x2), int(y2))
                    plate_corrected = _analyzer._perspective_correct(best_frame, quad) if quad is not None else None
        else:
            log.info("PlateRecognizer returned no result, falling back to local LPR")
            if _analyzer:
                _p, _c, plate_bbox, plate_corrected = _analyzer.read_plate(best_frame)
                plate = _normalize_plate(_p)
                conf = _c if plate else 0.0
    elif _analyzer:
        _p, _c, plate_bbox, plate_corrected = _analyzer.read_plate(best_frame)
        plate = _normalize_plate(_p)
        conf = _c if plate else 0.0

    log.info(f"LPR: plate={plate!r} conf={conf:.2f} bbox={plate_bbox}")

    # Save raw plate crop (4× upscale) and the perspective-corrected OCR crop
    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        fh, fw = best_frame.shape[:2]
        pad = max(8, int((py2 - py1) * 0.5))
        cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
        cx2, cy2 = min(fw, px2 + pad), min(fh, py2 + pad)
        plate_crop = best_frame[cy1:cy2, cx1:cx2]
        if plate_crop.size > 0:
            ph, pw = plate_crop.shape[:2]
            big = cv2.resize(plate_crop, (pw * 4, ph * 4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(event_dir, "plate_crop.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if plate_corrected is not None and plate_corrected.size > 0:
        enhanced = _analyzer._enhance_crop(plate_corrected)
        ch, cw = enhanced.shape[:2]
        if cw < 300:  # upscale small crops for display
            enhanced = cv2.resize(enhanced, (300, int(300 * ch / cw)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(event_dir, "plate_ocr.jpg"), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save thumbnail — vehicle crop if plate found, full frame otherwise
    snapshot_file = f"{event_id}.jpg"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_file)
    thumb = _vehicle_crop(best_frame, plate_bbox)
    # Cap thumbnail width to 1280px to keep file size reasonable
    th, tw = thumb.shape[:2]
    if tw > 1280:
        thumb = cv2.resize(thumb, (1280, int(th * 1280 / tw)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(snapshot_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # Also save the full best frame so event detail can display it
    cv2.imwrite(os.path.join(event_dir, "best_frame.jpg"), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

    from database import is_whitelisted
    if plate and is_whitelisted(plate):
        log.info(f"Skipped (whitelist): {plate} — event {event_id[:8]}")
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    hex_color, color_name = _vehicle_color(best_frame, plate_bbox)
    insert_event(
        event_id, camera_name, int(time.time()),
        f"snapshots/{snapshot_file}",
        f"events/{event_id}/clip.mp4",
        plate or None, hex_color, color_name,
    )
    log.info(f"Stored: {event_id[:8]} | plate={plate} | color={color_name} | frames={len(frame_files)}")
    return event_id


def _process_event():
    event_id = str(uuid.uuid4())
    event_dir = os.path.join(EVENTS_DIR, event_id)
    os.makedirs(event_dir, exist_ok=True)

    url = _rtsp_url()
    clip_path = os.path.join(event_dir, "clip.mp4")

    log.info(f"Capturing {CAPTURE_DURATION}s at {CAPTURE_FPS}fps — event {event_id[:8]}")

    ret = subprocess.run([
        "ffmpeg", "-y", "-rtsp_transport", "tcp",
        "-i", url, "-t", str(CAPTURE_DURATION), "-c", "copy", clip_path,
    ], capture_output=True, timeout=CAPTURE_DURATION + 10)

    if not os.path.exists(clip_path) or os.path.getsize(clip_path) < 1000:
        log.error(f"ffmpeg capture failed: {ret.stderr[-200:].decode(errors='ignore')}")
        shutil.rmtree(event_dir, ignore_errors=True)
        return

    _process_clip(event_id, event_dir, clip_path)


def process_uploaded_clip(src_path: str) -> str:
    """Create a new event from an uploaded MP4. Returns event_id."""
    event_id = str(uuid.uuid4())
    event_dir = os.path.join(EVENTS_DIR, event_id)
    os.makedirs(event_dir, exist_ok=True)
    clip_path = os.path.join(event_dir, "clip.mp4")
    shutil.copy2(src_path, clip_path)
    log.info(f"Processing uploaded clip — event {event_id[:8]}")
    result = _process_clip(event_id, event_dir, clip_path, camera_name="upload")
    if result is None:
        raise RuntimeError("Processing failed: no frames extracted")
    return event_id


def run_watcher():
    global _last_event_time, _analyzer

    os.makedirs(EVENTS_DIR, exist_ok=True)
    log.info("Loading plate analyzer...")
    _analyzer = PlateAnalyzer()

    log.info(f"Watcher started — polling {CAMERA_URL} every {POLL_INTERVAL}s")
    while True:
        try:
            state = _get_ai_state()
            if state:
                vehicle = state.get("vehicle", {})
                if vehicle.get("alarm_state") == 1:
                    now = time.time()
                    if now - _last_event_time > COOLDOWN:
                        _last_event_time = now
                        log.info("Vehicle detected!")
                        _process_event()
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            log.error(f"Watcher loop error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)
