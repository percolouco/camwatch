import os
import cv2
import numpy as np
import logging

log = logging.getLogger("lpr")

MODEL_CACHE = os.environ.get("MODEL_CACHE", "/models")

_YOLO_CONF_THRESHOLD = 0.04


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list, iou_threshold: float = 0.3) -> list:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -b[4])
    suppressed = [False] * len(boxes)
    result = []
    for i, b1 in enumerate(boxes):
        if suppressed[i]:
            continue
        result.append(b1)
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and _iou(b1, boxes[j]) > iou_threshold:
                suppressed[j] = True
    return result


def call_platerecognizer(frame: np.ndarray, api_key: str, region: str = "fr") -> tuple[str, float]:
    """Send a frame to PlateRecognizer API. Returns (plate_text, confidence)."""
    import requests as req

    h, w = frame.shape[:2]
    if w > 1920:
        scale = 1920 / w
        frame = cv2.resize(frame, (1920, int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        resp = req.post(
            "https://api.platerecognizer.com/v1/plate-reader/",
            headers={"Authorization": f"Token {api_key}"},
            files={"upload": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={"regions": region},
            timeout=15,
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            best = max(results, key=lambda r: r.get("score", 0))
            plate = best.get("plate", "").upper().strip()
            conf = float(best.get("score", 0))
            if len([c for c in plate if c.isalnum()]) >= 4:
                log.info(f"PlateRecognizer: {plate!r} conf={conf:.2f}")
                return plate, conf
    except Exception as e:
        log.warning(f"PlateRecognizer API error: {e}")
    return "", 0.0


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class PlateAnalyzer:
    def __init__(self):
        self._yolo = None
        self._rec = None
        self._keys: list[str] = []
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]

            yolo_path = os.path.join(MODEL_CACHE, "yolov9_license_plate", "yolov9-256-license-plates.onnx")
            rec_path = os.path.join(MODEL_CACHE, "paddleocr-onnx", "recognition_v4.onnx")
            keys_path = os.path.join(MODEL_CACHE, "paddleocr-onnx", "ppocr_keys_v1.txt")

            if os.path.exists(yolo_path):
                self._yolo = ort.InferenceSession(yolo_path, providers=providers)
                log.info("YOLOv9 plate detector loaded")
            else:
                log.warning(f"YOLOv9 model not found at {yolo_path}")

            if os.path.exists(rec_path):
                self._rec = ort.InferenceSession(rec_path, providers=providers)
                log.info("PaddleOCR recognition model loaded")

            if os.path.exists(keys_path):
                with open(keys_path) as f:
                    self._keys = f.read().splitlines()
                log.info(f"Loaded {len(self._keys)} OCR characters")

        except ImportError:
            log.warning("onnxruntime not installed — LPR disabled")
        except Exception as e:
            log.error(f"LPR model load error: {e}")

    def _yolo_on_tile(self, tile: np.ndarray, offset_x: int, offset_y: int) -> list[tuple]:
        th, tw = tile.shape[:2]
        inp = cv2.resize(tile, (256, 256))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]

        out = self._yolo.run(None, {"images": inp})[0]

        boxes = []
        for row in out:
            if len(row) < 7:
                continue
            _, x1, y1, x2, y2, _, conf = row
            if conf < _YOLO_CONF_THRESHOLD:
                continue
            x1 = offset_x + max(0.0, float(x1) / 256.0 * tw)
            y1 = offset_y + max(0.0, float(y1) / 256.0 * th)
            x2 = offset_x + min(float(tw), float(x2) / 256.0 * tw)
            y2 = offset_y + min(float(th), float(y2) / 256.0 * th)
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 4:
                continue
            ratio = bw / bh
            if ratio < 2.0 or ratio > 6.5:
                continue
            boxes.append((x1, y1, x2, y2, float(conf)))
        return boxes

    def detect_plates(self, frame: np.ndarray) -> list[tuple]:
        """Tiled YOLO detection — returns [(x1,y1,x2,y2,conf),...] in original coords."""
        if self._yolo is None:
            return []

        h, w = frame.shape[:2]
        all_boxes = []

        # Ignore top 15% (sky/trees) and bottom 8% (timestamp overlay)
        y_start = int(h * 0.15)
        y_end = int(h * 0.92)
        work = frame[y_start:y_end, :]
        wh = y_end - y_start

        cols, rows = 3, 2
        overlap = 0.2
        tw = int(w / (cols - overlap * (cols - 1)))
        th = int(wh / (rows - overlap * (rows - 1)))
        step_x = int(tw * (1 - overlap))
        step_y = int(th * (1 - overlap))

        for row in range(rows):
            for col in range(cols):
                tx1 = col * step_x
                ty1 = row * step_y
                tx2 = min(w, tx1 + tw)
                ty2 = min(wh, ty1 + th)
                tile = work[ty1:ty2, tx1:tx2]
                all_boxes.extend(self._yolo_on_tile(tile, tx1, ty1 + y_start))

        return _nms(all_boxes, iou_threshold=0.3)

    def _find_plate_quad(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
        """Find the 4 corners of the plate using minAreaRect on the white plate region."""
        fh, fw = frame.shape[:2]
        pw, ph = x2 - x1, y2 - y1

        mx = int(pw * 0.6)
        my = int(ph * 1.2)
        rx1 = max(0, x1 - mx)
        ry1 = max(0, y1 - my)
        rx2 = min(fw, x2 + mx)
        ry2 = min(fh, y2 + my)
        region = frame[ry1:ry2, rx1:rx2]

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Use the largest bright contour (plate background)
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < pw * ph * 0.3:
            return None

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] += rx1
        box[:, 1] += ry1

        # Validate: the oriented rect should be plate-shaped
        bw = rect[1][0]
        bh = rect[1][1]
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        if short_side < 4 or long_side / short_side < 2.0:
            return None

        return box

    def _perspective_correct(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Warp the detected plate quad to a frontal rectangle."""
        pts = _order_points(quad)
        # French plate: 520×110mm → 4.73:1
        target_h = 80
        target_w = int(target_h * 4.73)
        dst = np.array([
            [0, 0], [target_w - 1, 0],
            [target_w - 1, target_h - 1], [0, target_h - 1],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(frame, M, (target_w, target_h))

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """CLAHE contrast + sharpen for better OCR."""
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(crop, -1, kernel)

    def _ocr_tesseract(self, crop: np.ndarray) -> tuple[str, float]:
        try:
            import pytesseract
            # Skip EU blue strip (left ~11%) which confuses OCR
            eu_skip = max(0, int(crop.shape[1] * 0.11))
            crop = crop[:, eu_skip:]

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Scale to at least 3× for reliable Tesseract recognition
            if gray.shape[0] < 120:
                scale = max(2, 120 // gray.shape[0])
                gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                                  interpolation=cv2.INTER_CUBIC)

            wl = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            results = []
            for psm in [8, 7, 11]:
                cfg = f"--psm {psm} --oem 3 {wl}"
                t = pytesseract.image_to_string(gray, config=cfg).strip().upper()
                t = "".join(c for c in t if c.isalnum() or c == "-")
                alnum = "".join(c for c in t if c.isalnum())
                if len(alnum) >= 4:
                    results.append(t)
            if results:
                # Pick the result with the most alphanumeric characters
                best = max(results, key=lambda r: len([c for c in r if c.isalnum()]))
                return best, 0.6
        except Exception as e:
            log.debug(f"Tesseract error: {e}")
        return "", 0.0

    def _ocr_paddle(self, crop: np.ndarray) -> tuple[str, float]:
        if self._rec is None or not self._keys or crop.size == 0:
            return "", 0.0
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return "", 0.0
        inp_h = 48
        inp_w = max(10, int(inp_h * w / h))
        resized = cv2.resize(crop, (inp_w, inp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb / 127.5) - 1.0
        inp = normalized.transpose(2, 0, 1)[np.newaxis]
        out = self._rec.run(None, {"x": inp})[0][0]
        chars: list[str] = []
        confs: list[float] = []
        prev_idx = 0
        for step in out:
            idx = int(np.argmax(step))
            conf = float(step[idx])
            if idx != prev_idx and idx != 0 and idx <= len(self._keys):
                chars.append(self._keys[idx - 1])
                confs.append(conf)
            prev_idx = idx
        text = "".join(chars)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return text, avg_conf

    def _ocr_crop(self, crop: np.ndarray) -> tuple[str, float]:
        if crop.size == 0:
            return "", 0.0
        enhanced = self._enhance_crop(crop)
        text, conf = self._ocr_tesseract(enhanced)
        if text:
            return text, conf
        return self._ocr_paddle(enhanced)

    def read_plate(self, frame: np.ndarray) -> tuple[str, float, tuple | None, np.ndarray | None]:
        """Detect + read plate.
        Returns (plate_text, confidence, bbox_or_None, corrected_crop_or_None)."""
        plate_boxes = self.detect_plates(frame)
        if not plate_boxes:
            return "", 0.0, None, None

        x1, y1, x2, y2, plate_conf = plate_boxes[0]
        bbox = (int(x1), int(y1), int(x2), int(y2))

        # Try perspective correction first
        quad = self._find_plate_quad(frame, int(x1), int(y1), int(x2), int(y2))
        if quad is not None:
            corrected = self._perspective_correct(frame, quad)
            log.debug("Perspective correction applied")
        else:
            fh, fw = frame.shape[:2]
            pad = 8
            corrected = frame[max(0, int(y1) - pad):min(fh, int(y2) + pad),
                              max(0, int(x1) - pad):min(fw, int(x2) + pad)]

        text, ocr_conf = self._ocr_crop(corrected)

        alnum = "".join(c for c in text if c.isalnum())
        if len(alnum) < 4:
            return "", 0.0, bbox, corrected

        return text, (plate_conf + ocr_conf) / 2.0, bbox, corrected
