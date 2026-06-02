import os
import time
import uuid
import logging
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
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))
CAPTURE_DURATION = int(os.environ.get("CAPTURE_DURATION", "20"))
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
            log.debug("Camera login OK")
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


def _frame_score(frame: np.ndarray, plates: list) -> float:
    """Score a frame: prefer large, high-confidence plates. Fallback to sharpness."""
    if plates:
        x1, y1, x2, y2, conf = plates[0]
        return (x2 - x1) * (y2 - y1) * conf
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() * 0.001


def _capture_best_frame() -> np.ndarray | None:
    url = _rtsp_url()
    log.info(f"Capturing RTSP for {CAPTURE_DURATION}s...")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        log.error("Cannot open RTSP stream")
        return None

    best_frame: np.ndarray | None = None
    best_score = -1.0
    start = time.time()

    try:
        while time.time() - start < CAPTURE_DURATION:
            ret, frame = cap.read()
            if not ret:
                log.warning("RTSP read failed, retrying...")
                time.sleep(0.5)
                continue
            plates = _analyzer.detect_plates(frame) if _analyzer else []
            score = _frame_score(frame, plates)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
            time.sleep(0.8)
    finally:
        cap.release()

    log.info(f"Capture done. Best frame score: {best_score:.2f}")
    return best_frame


def _process_event():
    frame = _capture_best_frame()
    if frame is None:
        log.warning("No frame captured")
        return

    event_id = str(uuid.uuid4())
    snapshot_file = f"{event_id}.jpg"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_file)
    cv2.imwrite(snapshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    plate, conf = "", 0.0
    if _analyzer:
        plate, conf = _analyzer.read_plate(frame)

    log.info(f"LPR: plate={plate!r} conf={conf:.2f}")

    hex_color, color_name = extract_dominant_color_from_frame(frame)
    insert_event(
        event_id, CAMERA_NAME, int(time.time()),
        f"snapshots/{snapshot_file}", plate or None, hex_color, color_name,
    )
    log.info(f"Stored: {event_id} | plate={plate} | color={color_name}")


def run_watcher():
    global _last_event_time, _analyzer

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
                        log.info("Vehicle detected! Triggering capture...")
                        _process_event()
        except Exception as e:
            log.error(f"Watcher loop error: {e}", exc_info=True)
        time.sleep(POLL_INTERVAL)
