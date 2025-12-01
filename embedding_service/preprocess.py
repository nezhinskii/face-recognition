import numpy as np
import cv2
from typing import List, Dict, Any

def preprocess_image(img_bgr: np.ndarray, input_size=(112, 112)) -> np.ndarray:
    img = cv2.resize(img_bgr, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    img = img[np.newaxis, ...]
    return img

TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def extract_largest_face_aligned(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    output_size: int = 112
) -> np.ndarray | None:
    if not detections:
        return None

    areas = [(d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) for d in detections]
    best = detections[np.argmax(areas)]
    kps = np.array(best['keypoints']).reshape(5, 2).astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(kps, TEMPLATE_112, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        img,
        M,
        (output_size, output_size),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_LINEAR
    )
    return aligned