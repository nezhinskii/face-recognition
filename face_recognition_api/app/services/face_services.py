import requests
import json
import logging
from typing import List, Dict, Any
from app.config import settings


def detect_faces(image_bytes: bytes) -> List[Dict[str, Any]]:
    response = requests.post(
        f"{settings.detection_url}/detect",
        files={"image": image_bytes}
    )
    response.raise_for_status()
    return response.json()

logger = logging.getLogger(__name__)
def get_embedding(image_bytes: bytes, detections: List[Dict[str, Any]]) -> List[float]:
    response = requests.post(
        f"{settings.embedding_url}/embed",
        files={"image": image_bytes},
        data = {
            "detections": json.dumps(detections)
        }
    )
    response.raise_for_status()
    return response.json()