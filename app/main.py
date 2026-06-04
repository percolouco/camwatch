import os
import glob
import shutil
import logging
import threading
import tempfile
import asyncio
import subprocess
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


def _watcher_supervisor():
    while True:
        t = threading.Thread(target=watcher.run_watcher, daemon=True, name="watcher")
        t.start()
        t.join()
        logging.warning("Watcher thread exited unexpectedly — restarting in 5s")
        time.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    sup = threading.Thread(target=_watcher_supervisor, daemon=True, name="watcher-supervisor")
    sup.start()
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
    has_plate: str = Query(""),
    color: str = Query(""),
):
    limit = 20
    offset = (page - 1) * limit
    kw = dict(plate_filter=plate or None, camera_filter=camera or None,
              date_filter=date or None, has_plate=has_plate or None,
              color_filter=color or None)
    events = database.get_events(limit=limit, offset=offset, **kw)
    total  = database.count_events(**kw)
    cameras = database.get_cameras()
    colors  = database.get_distinct_colors()
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
        "colors": colors,
        "filter_plate": plate,
        "filter_camera": camera,
        "filter_date": date,
        "filter_has_plate": has_plate,
        "filter_color": color,
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

    # Full best frame (vehicle crop is only the thumbnail; full frame for detail view)
    best_frame_abs = os.path.join(event_dir, "best_frame.jpg")
    best_frame_path = f"events/{event_id}/best_frame.jpg" if os.path.exists(best_frame_abs) else ev.get("snapshot_path")

    is_wl = database.is_whitelisted(ev.get("plate") or "")
    return templates.TemplateResponse("event_detail.html", {
        "request": request,
        "ev": ev,
        "frames": frames,
        "has_clip": has_clip,
        "clip_size": clip_size,
        "plate_crop": plate_crop,
        "plate_ocr": plate_ocr,
        "best_frame_path": best_frame_path,
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


ZONE_PATH = "/data/zone.json"


@app.get("/config/zone", response_class=HTMLResponse)
async def zone_page(request: Request):
    import json
    zone: dict = {}
    if os.path.exists(ZONE_PATH):
        try:
            with open(ZONE_PATH) as f:
                zone = json.load(f)
        except Exception:
            pass

    latest_frame = None
    events = database.get_events(limit=1)
    if events:
        eid = events[0]["id"]
        bf = os.path.join(EVENTS_DIR, eid, "best_frame.jpg")
        if os.path.exists(bf):
            latest_frame = f"events/{eid}/best_frame.jpg"
        else:
            first = sorted(glob.glob(os.path.join(EVENTS_DIR, eid, "frame_*.jpg")))
            if first:
                latest_frame = f"events/{eid}/{os.path.basename(first[0])}"

    return templates.TemplateResponse("zone.html", {
        "request": request,
        "zone": zone,
        "latest_frame": latest_frame,
    })


@app.post("/config/zone")
async def save_zone(request: Request):
    import json
    data = await request.json()
    points = data.get("points", [])
    if len(points) == 0:
        if os.path.exists(ZONE_PATH):
            os.unlink(ZONE_PATH)
        return {"ok": True, "deleted": True}
    if len(points) < 3:
        raise HTTPException(400, "Minimum 3 points requis")
    with open(ZONE_PATH, "w") as f:
        json.dump({"points": points}, f)
    return {"ok": True, "points": len(points)}


@app.get("/config/snapshot")
async def live_snapshot():
    import tempfile, cv2
    url = watcher._rtsp_url()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name

    def _grab():
        subprocess.run([
            "ffmpeg", "-y", "-rtsp_transport", "tcp",
            "-i", url, "-vframes", "1", "-q:v", "2", tmp_path,
        ], capture_output=True, timeout=10)
        return os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0

    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, _grab)
    if ok:
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        from fastapi.responses import Response
        return Response(content=data, media_type="image/jpeg")
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    raise HTTPException(500, "Snapshot RTSP impossible")


@app.post("/event/{event_id}/test-lpr")
async def test_lpr(event_id: str, frame_path: str = Form(""), bbox: str = Form("")):
    import json, base64, cv2, numpy as np

    frame_abs = os.path.join("/data", frame_path) if frame_path else None
    if not frame_abs or not os.path.exists(frame_abs):
        raise HTTPException(404, "Frame introuvable")

    def _run():
        try:
            return _run_inner()
        except Exception as e:
            logging.exception("test-lpr unhandled error")
            return {"error": str(e)}

    def _run_inner():
        frame = cv2.imread(frame_abs)
        if frame is None:
            return {"error": "Impossible de lire l'image"}

        pr_key = watcher.PLATERECOGNIZER_KEY
        result: dict = {"has_pr": bool(pr_key)}

        if pr_key:
            from lpr import call_platerecognizer
            p, c = call_platerecognizer(frame, pr_key)
            if p:
                from watcher import _normalize_plate
                result["platerecognizer"] = {
                    "plate": _normalize_plate(p) or p,
                    "raw": p,
                    "conf": round(c, 3),
                }

        analyzer = watcher._analyzer
        if bbox and analyzer:
            try:
                box = json.loads(bbox)
                h, w = frame.shape[:2]
                cx2, cy2 = box["cx"] * w, box["cy"] * h
                bw2, bh2 = box["w"] * w, box["h"] * h
                x1 = max(0, int(cx2 - bw2 / 2))
                y1 = max(0, int(cy2 - bh2 / 2))
                x2 = min(w, int(cx2 + bw2 / 2))
                y2 = min(h, int(cy2 + bh2 / 2))

                source = "raw_crop"
                ocr_img = None
                quad = analyzer._find_plate_quad(frame, x1, y1, x2, y2)
                if quad is not None:
                    corrected = analyzer._perspective_correct(frame, quad)
                    if corrected is not None and corrected.size > 0:
                        ocr_img = analyzer._enhance_crop(corrected)
                        source = "perspective_corrected"

                if ocr_img is None:
                    crop = frame[y1:y2, x1:x2]
                    ocr_img = analyzer._enhance_crop(crop) if crop.size > 0 else None

                if ocr_img is not None:
                    text, conf = analyzer._ocr_paddle(ocr_img)
                    if not text:
                        text, conf = analyzer._ocr_tesseract(ocr_img)

                    oh, ow = ocr_img.shape[:2]
                    if ow > 0 and ow < 300:
                        scale = 300 / ow
                        ocr_display = cv2.resize(ocr_img, (300, int(oh * scale)), interpolation=cv2.INTER_CUBIC)
                    else:
                        ocr_display = ocr_img
                    _, buf = cv2.imencode(".jpg", ocr_display, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    from watcher import _normalize_plate
                    result["local"] = {
                        "plate": _normalize_plate(text) or text,
                        "raw": text,
                        "conf": round(conf, 3),
                        "img_b64": base64.b64encode(buf).decode(),
                        "source": source,
                    }
            except Exception as e:
                result["local_error"] = str(e)

        return result

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


@app.get("/health")
async def health():
    return {"status": "ok"}
