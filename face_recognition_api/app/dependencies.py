from fastapi import Depends
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from typing import Annotated, Generator

from app.database import SessionLocal
from app.config import settings

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

_qdrant_client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    timeout=10.0
)
def get_qdrant() -> QdrantClient:
    return _qdrant_client

DbSession = Annotated[Session, Depends(get_db)]
QdrantDep = Annotated[QdrantClient, Depends(get_qdrant)]