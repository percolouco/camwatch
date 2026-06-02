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


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def event_detail(request: Request, event_id: str):
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")

    ev["time_str"] = ts_to_str(ev["start_time"])

    # List frames for this event
    event_dir = os.path.join(EVENTS_DIR, event_id)
    frame_paths = sorted(glob.glob(os.path.join(event_dir, "frame_*.jpg")))
    frames = [f"events/{event_id}/frame_{i+1:04d}.jpg" for i in range(len(frame_paths))]

    has_clip = ev.get("clip_path") and os.path.exists(os.path.join("/data", ev["clip_path"]))

    return templates.TemplateResponse("event_detail.html", {
        "request": request,
        "ev": ev,
        "frames": frames,
        "has_clip": has_clip,
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
