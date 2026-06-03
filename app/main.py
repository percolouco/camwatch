import os
import glob
import shutil
import logging
import threading
import tempfile
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime

import database
import watcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

SNAPSHOTS_DIR = os.environ.get("SNAPSHOTS_DIR", "/data/snapshots")
EVENTS_DIR = os.environ.get("EVENTS_DIR", "/data/events")
ANNOTATIONS_DIR = os.environ.get("ANNOTATIONS_DIR", "/data/annotations")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(os.path.join(ANNOTATIONS_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(ANNOTATIONS_DIR, "labels"), exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    t = threading.Thread(target=watcher.run_watcher, daemon=True)
    t.start()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")
app.mount("/events", StaticFiles(directory=EVENTS_DIR), name="events")
app.mount("/annotations", StaticFiles(directory=ANNOTATIONS_DIR), name="annotations")

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

    plate_crop_abs = os.path.join(event_dir, "plate_crop.jpg")
    plate_crop = f"events/{event_id}/plate_crop.jpg" if os.path.exists(plate_crop_abs) else None
    plate_ocr_abs = os.path.join(event_dir, "plate_ocr.jpg")
    plate_ocr = f"events/{event_id}/plate_ocr.jpg" if os.path.exists(plate_ocr_abs) else None

    is_wl = database.is_whitelisted(ev.get("plate") or "")
    return templates.TemplateResponse("event_detail.html", {
        "request": request,
        "ev": ev,
        "frames": frames,
        "has_clip": has_clip,
        "clip_size": clip_size,
        "plate_crop": plate_crop,
        "plate_ocr": plate_ocr,
        "capture_duration": CAPTURE_DURATION,
        "capture_fps": CAPTURE_FPS,
        "capture_total": CAPTURE_DURATION * CAPTURE_FPS,
        "is_whitelisted": is_wl,
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


@app.post("/event/{event_id}/delete")
async def delete_event(event_id: str):
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")
    database.delete_event(event_id)
    event_dir = os.path.join(EVENTS_DIR, event_id)
    shutil.rmtree(event_dir, ignore_errors=True)
    snapshot = os.path.join(SNAPSHOTS_DIR, f"{event_id}.jpg")
    if os.path.exists(snapshot):
        os.unlink(snapshot)
    return RedirectResponse("/", status_code=303)


@app.post("/event/{event_id}/plate")
async def update_plate(event_id: str, plate: str = Form("")):
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")
    database.update_plate(event_id, plate)
    return RedirectResponse(f"/event/{event_id}", status_code=303)


@app.post("/upload")
async def upload_clip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .mp4 sont acceptés")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        event_id = await loop.run_in_executor(None, watcher.process_uploaded_clip, tmp_path)
    finally:
        os.unlink(tmp_path)

    return RedirectResponse(f"/event/{event_id}", status_code=303)


@app.get("/event/{event_id}/annotate", response_class=HTMLResponse)
async def annotate_page(request: Request, event_id: str):
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")
    ev["time_str"] = ts_to_str(ev["start_time"])
    event_dir = os.path.join(EVENTS_DIR, event_id)
    frame_paths = sorted(glob.glob(os.path.join(event_dir, "frame_*.jpg")))
    frames = [f"events/{event_id}/{os.path.basename(p)}" for p in frame_paths]
    ann_count = database.count_annotations()
    return templates.TemplateResponse("annotate.html", {
        "request": request,
        "ev": ev,
        "frames": frames,
        "ann_count": ann_count,
    })


@app.post("/event/{event_id}/annotate")
async def save_annotation(
    event_id: str,
    frame_path: str = Form(""),
    plate: str = Form(""),
    bbox: str = Form(""),
):
    import json, uuid
    ev = database.get_event(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Événement introuvable")
    plate = plate.strip().upper()
    if not plate or not frame_path or not bbox:
        raise HTTPException(status_code=400, detail="Données manquantes")
    try:
        box = json.loads(bbox)
        cx, cy, bw, bh = float(box["cx"]), float(box["cy"]), float(box["w"]), float(box["h"])
    except Exception:
        raise HTTPException(status_code=400, detail="bbox invalide")

    ann_id = str(uuid.uuid4())
    src = os.path.join("/data", frame_path)
    dst_img = os.path.join(ANNOTATIONS_DIR, "images", f"{ann_id}.jpg")
    shutil.copy2(src, dst_img)
    with open(os.path.join(ANNOTATIONS_DIR, "labels", f"{ann_id}.txt"), "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    csv_path = os.path.join(ANNOTATIONS_DIR, "plates.csv")
    with open(csv_path, "a") as f:
        f.write(f"{ann_id}.jpg,{plate}\n")
    database.save_annotation(ann_id, event_id, frame_path, plate, cx, cy, bw, bh)
    return RedirectResponse(f"/event/{event_id}?annotated=1", status_code=303)


@app.get("/dataset/export")
async def export_annotations():
    import zipfile, io
    from fastapi.responses import StreamingResponse
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        img_dir = os.path.join(ANNOTATIONS_DIR, "images")
        lbl_dir = os.path.join(ANNOTATIONS_DIR, "labels")
        csv_path = os.path.join(ANNOTATIONS_DIR, "plates.csv")
        for f in glob.glob(os.path.join(img_dir, "*.jpg")):
            zf.write(f, f"images/{os.path.basename(f)}")
        for f in glob.glob(os.path.join(lbl_dir, "*.txt")):
            zf.write(f, f"labels/{os.path.basename(f)}")
        if os.path.exists(csv_path):
            zf.write(csv_path, "plates.csv")
        # data.yaml for YOLO training
        yaml_content = "path: .\ntrain: images\nval: images\nnc: 1\nnames: ['license_plate']\n"
        zf.writestr("data.yaml", yaml_content)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=camwatch_dataset.zip"},
    )


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    rows = database.get_plate_stats()
    whitelist_plates = {w["plate"] for w in database.get_whitelist()}
    for r in rows:
        r["first_str"] = ts_to_str(r["first_seen"])
        r["last_str"] = ts_to_str(r["last_seen"])
        r["whitelisted"] = r["plate"] in whitelist_plates
    return templates.TemplateResponse("stats.html", {"request": request, "stats": rows})


@app.get("/whitelist", response_class=HTMLResponse)
async def whitelist_page(request: Request):
    entries = database.get_whitelist()
    for e in entries:
        e["added_str"] = ts_to_str(e["added_at"])
    return templates.TemplateResponse("whitelist.html", {"request": request, "entries": entries})


@app.post("/whitelist/add")
async def whitelist_add(plate: str = Form(""), label: str = Form(""), back: str = Form("")):
    plate = plate.strip().upper()
    if plate:
        database.add_to_whitelist(plate, label)
    return RedirectResponse(back or "/whitelist", status_code=303)


@app.post("/whitelist/remove/{plate}")
async def whitelist_remove(plate: str, back: str = Form("")):
    database.remove_from_whitelist(plate)
    return RedirectResponse(back or "/whitelist", status_code=303)


@app.get("/health")
async def health():
    return {"status": "ok"}
