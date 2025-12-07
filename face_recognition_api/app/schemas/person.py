from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    bbox: List[int]
    keypoints: List[int]
    conf: float

class NewPersonResponse(BaseModel):
    status: str = "success"
    name: str
    qdrant_id: str
    faces_detected: int
    best_det_id: int
    detections: List[Detection]

class GetPersonRequest(BaseModel):
    threshold: float = 0.40
    model_config = {"extra": "ignore"}

class GetPersonResponse(BaseModel):
    name: str | None = None
    similarity: float = 0.0
    faces_detected: int
    best_det_id: int
    detections: List[Detection]