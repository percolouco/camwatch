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

_token: str | None = None
_token_time = 0.0
_last_event_time = 0.0
_analyzer: PlateAnalyzer | None = None


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

    # Score ALL frames, rename sorted (best = frame_0001)
    scored: list[tuple[float, np.ndarray, str]] = []
    for path in frame_files:
        frame = cv2.imread(path)
        if frame is None:
            continue
        plates = _analyzer.detect_plates(frame) if _analyzer else []
        score = _frame_score(frame, plates)
        scored.append((score, frame, path))

    scored.sort(key=lambda x: -x[0])

    for i, (_, _, old_path) in enumerate(scored):
        os.rename(old_path, old_path + ".tmp")
    for i, (_, _, old_path) in enumerate(scored):
        os.rename(old_path + ".tmp", os.path.join(event_dir, f"frame_{i+1:04d}.jpg"))

    best_frame = scored[0][1] if scored else None
    if best_frame is None:
        shutil.rmtree(event_dir, ignore_errors=True)
        return None

    # LPR on best frame
    plate, conf, plate_bbox = ("", 0.0, None)
    if _analyzer:
        plate, conf, plate_bbox = _analyzer.read_plate(best_frame)
    log.info(f"LPR: plate={plate!r} conf={conf:.2f} bbox={plate_bbox}")

    # Save plate crop for manual review (even when OCR fails)
    if plate_bbox:
        px1, py1, px2, py2 = plate_bbox
        fh, fw = best_frame.shape[:2]
        pad = max(8, int((py2 - py1) * 0.5))
        cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
        cx2, cy2 = min(fw, px2 + pad), min(fh, py2 + pad)
        plate_crop = best_frame[cy1:cy2, cx1:cx2]
        if plate_crop.size > 0:
            # Save at 4× upscale for readability
            ph, pw = plate_crop.shape[:2]
            big = cv2.resize(plate_crop, (pw * 4, ph * 4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(event_dir, "plate_crop.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save thumbnail
    snapshot_file = f"{event_id}.jpg"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_file)
    cv2.imwrite(snapshot_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

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
        except Exception as e:
            log.error(f"Watcher loop error: {e}", exc_info=True)
        time.sleep(POLL_INTERVAL)
