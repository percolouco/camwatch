import os
import glob
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime

import database
import watcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SNAPSHOTS_DIR = os.environ.get("SNAPSHOTS_DIR", "/data/snapshots")
EVENTS_DIR = os.environ.get("EVENTS_DIR", "/data/events")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    t = threading.Thread(target=watcher.run_watcher, daemon=True)
    t.start()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")
app.mount("/events", StaticFiles(directory=EVENTS_DIR), name="events")

templates = Jinja2Templates(directory="/app/templates")


def ts_to_str(ts):
    try:
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return str(ts)


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    page: int = Query(1, ge=1),
    plate: str = Query(""),
    camera: str = Query(""),
    date: str = Query(""),
):
    limit = 20
    offset = (page - 1) * limit
    events = database.get_events(limit=limit, offset=offset,
                                  plate_filter=plate or None,
                                  camera_filter=camera or None,
                                  date_filter=date or None)
    total = database.count_events(plate_filter=plate or None,
                                   camera_filter=camera or None,
                                   date_filter=date or None)
    cameras = database.get_cameras()
    total_pages = max(1, (total + limit - 1) // limit)

    for ev in events:
        ev["time_str"] = ts_to_str(ev["start_time"])

    return templates.TemplateResponse("index.html", {
        "request": request,
        "events": events,
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "cameras": cameras,
        "filter_plate": plate,
        "filter_camera": camera,
        "filter_date": date,
    })


CAPTURE_DURATION = int(os.environ.get("CAPTURE_DURATION", "15"))
CAPTURE_FPS = int(os.environ.get("CAPTURE_FPS", "5"))
TOP_FRAMES = int(os.environ.get("TOP_FRAMES", "10"))


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: str):
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")

    ev["time_str"] = ts_to_str(ev["start_time"])

    event_dir = os.path.join(EVENTS_DIR, event_id)
    frame_paths = sorted(glob.glob(os.path.join(event_dir, "frame_*.jpg")))
    frames = [f"events/{event_id}/{os.path.basename(p)}" for p in frame_paths]

    clip_abs = os.path.join("/data", ev["clip_path"]) if ev.get("clip_path") else None
    has_clip = bool(clip_abs and os.path.exists(clip_abs))
    clip_size = ""
    if has_clip:
        size = os.path.getsize(clip_abs)
        clip_size = f"{size // 1024 // 1024}MB" if size > 1024 * 1024 else f"{size // 1024}KB"

    return templates.TemplateResponse("event_detail.html", {
        "request": request,
        "ev": ev,
        "frames": frames,
        "has_clip": has_clip,
        "clip_size": clip_size,
        "capture_duration": CAPTURE_DURATION,
        "capture_total": CAPTURE_DURATION * CAPTURE_FPS,
        "top_frames": TOP_FRAMES,
    })


@app.get("/api/events")
async def api_events(
    page: int = Query(1, ge=1),
    plate: str = Query(""),
    camera: str = Query(""),
    date: str = Query(""),
):
    limit = 20
    offset = (page - 1) * limit
    events = database.get_events(limit=limit, offset=offset,
                                  plate_filter=plate or None,
                                  camera_filter=camera or None,
                                  date_filter=date or None)
    total = database.count_events(plate_filter=plate or None,
                                   camera_filter=camera or None,
                                   date_filter=date or None)
    for ev in events:
        ev["time_str"] = ts_to_str(ev["start_time"])
    return {"events": events, "total": total}


@app.get("/health")
async def health():
    return {"status": "ok"}
