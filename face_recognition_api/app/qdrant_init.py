from qdrant_client.http.models import VectorParams, Distance
from app.config import settings
from app.dependencies import get_qdrant

def init_qdrant_collection() -> None:
    client = get_qdrant()
    collection_name = settings.qdrant_collection
    
    try:
        client.get_collection(collection_name)
    except Exception as e:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE
            )
        )