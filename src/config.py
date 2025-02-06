from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    PERSIST_DIRECTORY: str = "./chroma_db"
    OUTPUT_PATH: str = "./pdf_extracted_content/"
    BATCH_SIZE: int = 5
    RATE_LIMIT_DELAY: int = 2
    MAX_RETRIES: int = 3
    
    class Config:
        env_file = ".env"

settings = Settings() 