from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    bbox: List[int]
    keypoints: List[int]
    conf: float

class PersonResponse(BaseModel):
    id: int 
    name: str
    similarity: Optional[float] = None
    faces_detected: int
    best_det_id: int
    detections: List[Detection]