from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pydantic import validator

class Settings(BaseSettings):
    """Application settings with validation"""
    # API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    GROQ_API_KEY: Optional[str] = Field(None, description="Groq API key")
    
    # File paths
    DATA_DIR: str = Field("./data", description="Data directory")
    PDF_FILE: str = Field("attention_paper.pdf", description="PDF file name")
    PERSIST_DIRECTORY: str = Field("./chroma_db", description="Vector store persistence directory")
    OUTPUT_PATH: str = Field("./pdf_extracted_content/", description="Path for extracted content")
    
    # Processing settings
    BATCH_SIZE: int = Field(5, description="Batch size for processing")
    RATE_LIMIT_DELAY: int = Field(2, description="Delay between API calls")
    MAX_RETRIES: int = Field(3, description="Maximum number of retries")
    
    # Model settings
    MODEL_NAME: str = Field("gpt-4o-mini-2024-07-18", description="OpenAI model name")
    EMBEDDING_MODEL: str = Field("text-embedding-ada-002", description="Embedding model name")
    
    @property
    def pdf_path(self) -> str:
        return os.path.join(self.DATA_DIR, self.PDF_FILE)
    
    @validator("OPENAI_API_KEY")
    def validate_openai_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        return v
    
    @validator("PERSIST_DIRECTORY", "OUTPUT_PATH", "DATA_DIR")
    def create_directory(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings() 