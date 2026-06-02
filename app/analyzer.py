import cv2
import numpy as np
import colorsys


def extract_dominant_color_from_frame(frame: np.ndarray, bbox: dict | None = None) -> tuple[str, str]:
    """Returns (hex_color, color_name) from a BGR numpy frame."""
    if frame is None:
        return "#808080", "Inconnu"
    return _dominant_color(frame, bbox)


def extract_dominant_color(image_path: str, bbox: dict | None = None) -> tuple[str, str]:
    """Returns (hex_color, color_name) from image path, optionally cropped to bbox."""
    img = cv2.imread(image_path)
    if img is None:
        return "#808080", "Inconnu"
    return _dominant_color(img, bbox)


def _dominant_color(img: np.ndarray, bbox: dict | None = None) -> tuple[str, str]:

    if bbox:
        h, w = img.shape[:2]
        x1 = max(0, int(bbox.get("x", 0) * w))
        y1 = max(0, int(bbox.get("y", 0) * h))
        x2 = min(w, int((bbox.get("x", 0) + bbox.get("width", 1)) * w))
        y2 = min(h, int((bbox.get("y", 0) + bbox.get("height", 1)) * h))
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]

    small = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    k = min(4, len(pixels))
    _, labels, centers = cv2.kmeans(
        pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    b, g, r = int(dominant[0]), int(dominant[1]), int(dominant[2])
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return hex_color, _name_color(r, g, b)


def _name_color(r: int, g: int, b: int) -> str:
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if v < 0.18:
        return "Noir"
    if v > 0.82 and s < 0.18:
        return "Blanc"
    if s < 0.18:
        return "Gris"
    hue = h * 360
    if hue < 15 or hue >= 345:
        return "Rouge"
    if hue < 45:
        return "Orange"
    if hue < 75:
        return "Jaune"
    if hue < 150:
        return "Vert"
    if hue < 195:
        return "Cyan"
    if hue < 255:
        return "Bleu"
    if hue < 290:
        return "Violet"
    return "Rose"
