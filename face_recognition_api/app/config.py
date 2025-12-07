from pydantic_settings import BaseSettings, SettingsConfigDict
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
env_file_path = os.path.join(current_dir, "../../.env")

class Settings(BaseSettings):
    detection_url: str = "http://detection-service:3000"
    embedding_url: str = "http://embedding-service:3000"

    postgres_host: str
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: str
    postgres_db: str

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "face_embeddings"

    embedding_dim: int = 512

    def get_database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    model_config = SettingsConfigDict(env_file=env_file_path)

settings = Settings()