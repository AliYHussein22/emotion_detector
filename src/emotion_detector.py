"""
src/emotion_detector.py
───────────────────────
Wraps the FER (Facial Emotion Recognition) library.
FER uses a CNN trained on FER-2013 and a Haar-cascade / MTCNN face detector.

Detected emotions: angry · disgust · fear · happy · sad · surprise · neutral
"""

from __future__ import annotations
import numpy as np
import cv2

# Lazy import so Streamlit can start even if FER isn't installed yet
_fer = None

def _get_fer():
    global _fer
    if _fer is None:
        from fer import FER
        # mtcnn=True gives better accuracy; set False for speed on slow machines
        _fer = FER(mtcnn=False)
    return _fer


def detect_emotion(
    img_rgb: np.ndarray,
    confidence_threshold: float = 0.4,
) -> dict | None:
    """
    Run emotion detection on an RGB image.

    Returns
    -------
    dict with keys:
        emotion    : str   – dominant emotion label
        confidence : float – probability (0–1)
        box        : tuple – (x, y, w, h) bounding box in pixels
    or None if no face is found above the threshold.
    """
    detector = _get_fer()
    results  = detector.detect_emotions(img_rgb)

    if not results:
        return None

    # Pick the face with the highest dominant-emotion confidence
    best = None
    best_score = -1.0

    for face in results:
        emotions = face["emotions"]           # {"happy": 0.97, "sad": 0.01, …}
        dominant = max(emotions, key=emotions.get)
        score    = emotions[dominant]

        if score > best_score:
            best_score = score
            best = {
                "emotion":    dominant,
                "confidence": score,
                "box":        face["box"],    # (x, y, w, h)
                "all":        emotions,
            }

    if best is None or best["confidence"] < confidence_threshold:
        return None

    return best


def blur_background(
    img_rgb: np.ndarray,
    box: tuple,
    blur_strength: int = 31,
    padding: float = 0.35,
) -> np.ndarray:
    """
    Blur everything outside the face bounding box.

    Parameters
    ----------
    img_rgb        : RGB image as numpy array
    box            : (x, y, w, h) face bounding box from FER
    blur_strength  : kernel size for Gaussian blur (must be odd)
    padding        : fractional padding around the face box to keep sharp

    Returns
    -------
    New RGB image with background blurred.
    """
    x, y, w, h = box
    H, W = img_rgb.shape[:2]

    # Expand the face region with padding so we don't clip the face edges
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    # Ensure blur kernel is odd
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

    # Blur the full image
    blurred = cv2.GaussianBlur(img_rgb, (k, k), 0)

    # Create a smooth mask: white (1) inside face region, black (0) outside
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0

    # Feather the mask edges for a natural transition
    feather = max(15, int(min(w, h) * 0.15))
    feather = feather if feather % 2 == 1 else feather + 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask = mask[:, :, np.newaxis]  # shape (H, W, 1) for broadcasting

    # Blend: sharp face + blurred background
    output = (img_rgb * mask + blurred * (1 - mask)).astype(np.uint8)
    return output
