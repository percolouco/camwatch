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
        """Run YOLO on a single tile, return boxes in original frame coordinates."""
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
            # Plates are always wider than tall — French plates ~4.7:1
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

        # 3×2 tiles with 20% overlap so plates near tile edges are caught
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
                # offset_y accounts for the cropped top strip
                all_boxes.extend(self._yolo_on_tile(tile, tx1, ty1 + y_start))

        return _nms(all_boxes, iou_threshold=0.3)

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """Upscale + CLAHE + sharpen a plate crop for better OCR."""
        h, w = crop.shape[:2]
        # Upscale so the plate is at least 80px tall
        target_h = 80
        if h < target_h:
            scale = target_h / h
            crop = cv2.resize(crop, (max(10, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # Mild sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(crop, -1, kernel)

    def _ocr_tesseract(self, crop: np.ndarray) -> tuple[str, float]:
        """Tesseract OCR tuned for license plates."""
        try:
            import pytesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Try both single-word and single-line modes, take the longer result
            cfg = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text8 = pytesseract.image_to_string(gray, config=cfg).strip().replace(" ", "").upper()
            cfg7 = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text7 = pytesseract.image_to_string(gray, config=cfg7).strip().replace(" ", "").upper()
            text = text8 if len(text8) >= len(text7) else text7
            alnum = "".join(c for c in text if c.isalnum())
            if len(alnum) >= 4:
                return text, 0.6  # Tesseract doesn't give per-char conf easily; use fixed score
        except Exception as e:
            log.debug(f"Tesseract error: {e}")
        return "", 0.0

    def _ocr_paddle(self, crop: np.ndarray) -> tuple[str, float]:
        """PaddleOCR recognition fallback."""
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
        """Run OCR on a plate crop. Tesseract first, PaddleOCR as fallback."""
        if crop.size == 0:
            return "", 0.0
        enhanced = self._enhance_crop(crop)
        # Tesseract is better for Latin/French plates
        text, conf = self._ocr_tesseract(enhanced)
        if text:
            return text, conf
        # Fallback to PaddleOCR
        return self._ocr_paddle(enhanced)

    def read_plate(self, frame: np.ndarray) -> tuple[str, float, tuple | None]:
        """Detect plate, read text. Returns (plate_text, confidence, bbox_or_None)."""
        plate_boxes = self.detect_plates(frame)
        if not plate_boxes:
            return "", 0.0, None

        x1, y1, x2, y2, plate_conf = plate_boxes[0]
        pad = 8
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(frame.shape[1], int(x2) + pad)
        cy2 = min(frame.shape[0], int(y2) + pad)

        crop = frame[cy1:cy2, cx1:cx2]
        text, ocr_conf = self._ocr_crop(crop)

        alnum = "".join(c for c in text if c.isalnum())
        if len(alnum) < 4:
            return "", 0.0, (int(x1), int(y1), int(x2), int(y2))

        return text, (plate_conf + ocr_conf) / 2.0, (int(x1), int(y1), int(x2), int(y2))
