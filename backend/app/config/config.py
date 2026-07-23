import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY", "")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY", "")
    
    # App Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Vector DB / Embeddings
    VECTOR_DB_TYPE: str = "faiss"  # "faiss" or "chromadb"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Caching
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    USE_REDIS: bool = False  # Set to True if Redis is installed/active

    class Config:
        env_file = ".env"

settings = Settings()
