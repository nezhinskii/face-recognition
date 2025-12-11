from sqlalchemy import Column, Integer, Text, BigInteger, DateTime, UUID, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False, index=True)
    qdrant_id = Column(UUID, unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())