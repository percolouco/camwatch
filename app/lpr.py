import os
import cv2
import numpy as np
import logging

log = logging.getLogger("lpr")

MODEL_CACHE = os.environ.get("MODEL_CACHE", "/models")

# Output format: [N, 7] where each row = [batch_idx, x1, y1, x2, y2, class_id, confidence]
_YOLO_CONF_THRESHOLD = 0.03


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

    def detect_plates(self, frame: np.ndarray) -> list[tuple]:
        """Returns [(x1, y1, x2, y2, conf), ...] in original frame coordinates."""
        if self._yolo is None:
            return []

        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (256, 256))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]

        out = self._yolo.run(None, {"images": inp})[0]  # [N, 7]

        boxes = []
        for row in out:
            # row: [batch_idx, x1, y1, x2, y2, class_id, confidence]
            if len(row) >= 7:
                _, x1, y1, x2, y2, _, conf = row
                if conf < _YOLO_CONF_THRESHOLD:
                    continue
                # Scale from 256x256 to original
                x1 = max(0.0, float(x1) / 256.0 * w)
                y1 = max(0.0, float(y1) / 256.0 * h)
                x2 = min(float(w), float(x2) / 256.0 * w)
                y2 = min(float(h), float(y2) / 256.0 * h)
                if x2 > x1 + 4 and y2 > y1 + 4:
                    boxes.append((x1, y1, x2, y2, float(conf)))

        return sorted(boxes, key=lambda b: -b[4])

    def _ocr_crop(self, crop: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR recognition on a crop. Returns (text, confidence)."""
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

        out = self._rec.run(None, {"x": inp})[0][0]  # [seq, 6625]

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

    def read_plate(self, frame: np.ndarray) -> tuple[str, float, tuple | None]:
        """Detect plate, read text. Returns (plate_text, confidence, bbox_or_None).
        bbox = (x1, y1, x2, y2) in pixel coords."""
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

        # Filter noise: plate must have at least 4 alphanumeric chars
        alnum = "".join(c for c in text if c.isalnum())
        if len(alnum) < 4:
            return "", 0.0, (int(x1), int(y1), int(x2), int(y2))

        return text, (plate_conf + ocr_conf) / 2.0, (int(x1), int(y1), int(x2), int(y2))
