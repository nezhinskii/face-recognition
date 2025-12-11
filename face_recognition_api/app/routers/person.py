from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.dependencies import DbSession, QdrantDep
from app.repositories.person_repo import delete_person_by_id, get_person_by_name, create_person, get_person_by_qdrant_id, \
    search_similar_face, upsert_embedding, get_person_by_id, delete_from_qdrant
from app.services.face_services import detect_faces, get_embedding
from app.schemas.person_schemas import Detection, PersonResponse

router = APIRouter(prefix="/api", tags=["persons"])

SIMILARITY_THRESHOLD = 0.35

@router.post("/new_person", response_model=PersonResponse, status_code=201)
async def new_person(
    db: DbSession,
    qdrant: QdrantDep,
    name: str = Form(..., description="Person name"),
    file: UploadFile = File(..., media_type="image/*"),
):
    if get_person_by_name(db, name):
        raise HTTPException(409, detail=f"Person '{name}' already exists")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, detail="Empty file")

    detections = detect_faces(image_bytes)
    if not detections:
        raise HTTPException(422, detail="No faces found")

    embedding_result = get_embedding(image_bytes, detections)
    embedding = embedding_result["embedding"]
    best_det_id = embedding_result["best_det_id"]

    exists_qdrant_id, similarity = search_similar_face(qdrant, embedding, threshold=SIMILARITY_THRESHOLD)
    if exists_qdrant_id is not None:
        raise HTTPException(409, detail=f"Similar person already exists")

    qdrant_id = upsert_embedding(qdrant, name, embedding)
    person = create_person(db, name, qdrant_id)

    return PersonResponse(
        id=person.id,
        name=name,
        faces_detected=len(detections),
        best_det_id=best_det_id,
        detections=[Detection(**d) for d in detections],
    )


@router.post("/get_person", response_model=PersonResponse)
async def get_person(
    db: DbSession,
    qdrant: QdrantDep,
    file: UploadFile = File(..., media_type="image/*"),
    threshold: float = Form(SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Min similarity (0.0–1.0)"),
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

    qdrant_id, similarity = search_similar_face(qdrant, embedding, threshold=threshold)

    if not qdrant_id or not similarity:
        raise HTTPException(404, detail="Person not found")
    
    person = get_person_by_qdrant_id(db, qdrant_id)
    if not person:
        raise HTTPException(500, detail="Inconsistency: name found in Qdrant but not in Postgres")

    return PersonResponse(
        id=person.id,
        name=person.name,
        similarity=round(similarity, 4),
        faces_detected=len(detections),
        best_det_id=best_det_id,
        detections=[Detection(**d) for d in detections],
    )

@router.delete("/delete_person", status_code=204)
async def delete_person(
    db: DbSession,
    qdrant: QdrantDep,
    id: int = Form(..., description="Person id")
):
    person = get_person_by_id(db, id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        delete_from_qdrant(qdrant, str(person.qdrant_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete person")
    success = delete_person_by_id(db, id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete person")
    return