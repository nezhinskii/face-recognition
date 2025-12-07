from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from uuid import uuid4
from typing import List

from app.models.person import Person
from app.config import settings

def get_person_by_name(db: Session, name: str) -> Person | None:
    return db.query(Person).filter(Person.name == name).first()

def create_person(db: Session, name: str, qdrant_id: str) -> Person:
    db_person = Person(name=name, qdrant_id=qdrant_id)
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person

def upsert_embedding(qdrant: QdrantClient, name: str, embedding: List[float]):
    collection = settings.qdrant_collection

    point_id = str(uuid4())
    qdrant.upsert(
        collection_name=collection,
        points=[PointStruct(id=point_id, vector=embedding, payload={"name": name})]
    )
    return point_id

def search_similar_face(
    qdrant: QdrantClient,
    embedding: List[float],
    threshold: float = 0.40,
):
    collection = settings.qdrant_collection
    print(threshold)
    search_result = qdrant.search(
        collection_name=collection,
        query_vector=embedding,
        limit=1,
        score_threshold=threshold,
        with_payload=True,
    )

    if not search_result:
        return None, 0.0

    hit = search_result[0]
    return hit.payload["name"], hit.score