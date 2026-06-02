import os
import time
import logging
import requests
import shutil
from database import event_exists, insert_event
from analyzer import extract_dominant_color

log = logging.getLogger("watcher")

FRIGATE_URL = os.environ.get("FRIGATE_URL", "http://frigate:5000")
SNAPSHOTS_DIR = os.environ.get("SNAPSHOTS_DIR", "/data/snapshots")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "30"))


def fetch_new_events() -> list[dict]:
    try:
        resp = requests.get(
            f"{FRIGATE_URL}/api/events",
            params={"labels": "car", "has_snapshot": "1", "limit": "50"},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"Frigate fetch error: {e}")
        return []


def download_snapshot(event_id: str, dest_path: str) -> bool:
    try:
        resp = requests.get(
            f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg",
            params={"bbox": "1", "crop": "1", "quality": "95"},
            timeout=15, stream=True
        )
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
        return True
    except Exception as e:
        log.warning(f"Snapshot download error for {event_id}: {e}")
        return False


def process_event(ev: dict):
    event_id = ev.get("id", "")
    if not event_id or event_exists(event_id):
        return

    camera = ev.get("camera", "unknown")
    start_time = int(ev.get("start_time", time.time()))

    # Plate from Frigate LPR
    plate = None
    data = ev.get("data", {})
    if data.get("sub_label"):
        plate = data["sub_label"]
        if isinstance(plate, list):
            plate = plate[0] if plate else None

    # Download snapshot
    snapshot_file = f"{event_id}.jpg"
    snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_file)
    if not download_snapshot(event_id, snapshot_path):
        return

    # Extract color from vehicle bounding box
    bbox = None
    if ev.get("box"):
        b = ev["box"]
        bbox = {"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1]}
    elif data.get("box"):
        b = data["box"]
        if len(b) == 4:
            bbox = {"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1]}

    hex_color, color_name = extract_dominant_color(snapshot_path, bbox)

    insert_event(event_id, camera, start_time, f"snapshots/{snapshot_file}",
                 plate, hex_color, color_name)
    log.info(f"Processed event {event_id} | plate={plate} | color={color_name} | camera={camera}")


def run_watcher():
    log.info(f"Watcher started — polling Frigate every {POLL_INTERVAL}s")
    while True:
        events = fetch_new_events()
        for ev in events:
            try:
                process_event(ev)
            except Exception as e:
                log.error(f"Error processing event: {e}")
        time.sleep(POLL_INTERVAL)
