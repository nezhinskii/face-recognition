from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.dependencies import DbSession, QdrantDep
from app.repositories.person_repo import get_person_by_name, create_person, search_similar_face, upsert_embedding
from app.services.face_services import detect_faces, get_embedding
from app.schemas.person import GetPersonResponse, NewPersonResponse, Detection

router = APIRouter(prefix="/api", tags=["persons"])


@router.post("/new_person", response_model=NewPersonResponse, status_code=201)
async def new_person(
    db: DbSession,
    qdrant: QdrantDep,
    name: str = Form(..., description="Имя человека"),
    file: UploadFile = File(..., media_type="image/*"),
):
    if get_person_by_name(db, name):
        raise HTTPException(400, detail=f"Person '{name}' already exists")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, detail="Empty file")

    detections = detect_faces(image_bytes)
    if not detections:
        raise HTTPException(422, detail="No faces found")

    embedding_result = get_embedding(image_bytes, detections)
    embedding = embedding_result["embedding"]
    best_det_id = embedding_result["best_det_id"]

    qdrant_id = upsert_embedding(qdrant, name, embedding)
    create_person(db, name, qdrant_id)

    return NewPersonResponse(
        name=name,
        qdrant_id=qdrant_id,
        faces_detected=len(detections),
        best_det_id=best_det_id,
        detections=[Detection(**d) for d in detections]
    )


@router.post("/get_person", response_model=GetPersonResponse)
async def get_person(
    qdrant: QdrantDep,
    file: UploadFile = File(..., media_type="image/*"),
    threshold: float = Form(0.40, ge=0.0, le=1.0, description="Min similarity (0.0–1.0)"),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, detail="Empty file")

    detections = detect_faces(image_bytes)
    if not detections:
        raise HTTPException(422, detail="No faces found")

    result = get_embedding(image_bytes, detections)
    embedding = result["embedding"]
    best_det_id = result["best_det_id"]

    name, similarity = search_similar_face(qdrant, embedding, threshold=threshold)

    return GetPersonResponse(
        name=name,
        similarity=round(similarity, 4) if similarity else 0.0,
        faces_detected=len(detections),
        best_det_id=best_det_id,
        detections=[Detection(**d) for d in detections]
    )