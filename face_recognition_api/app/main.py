from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.models.person_model import Base
from app.routers.person import router as person_router
from app.database import engine
from app.qdrant_init import init_qdrant_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("face-api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Face Recognition API...")
    logger.info("Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")

    logger.info("Initializing Qdrant collection if not exist...")
    init_qdrant_collection()
    logger.info("Qdrant ready")
    yield
    logger.info("Shutting down Face Recognition API...")

app = FastAPI(
    title="Face Recognition API",
    description="Add and search people by face photo",
    version="1.0",
    lifespan=lifespan
)

app.include_router(person_router)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "API is running"}