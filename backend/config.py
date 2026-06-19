import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    api_key: Optional[str] = Field(default=None, alias="API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    alpha_vantage_key: Optional[str] = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    db_path: str = Field(default="chatbot.db", alias="DB_PATH")
    postgres_url: Optional[str] = Field(default=None, alias="POSTGRES_URL")
    chroma_host: Optional[str] = Field(default=None, alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True
    }

    # Ensure keys fall back to whichever is available
    def model_post_init(self, __context):
        if not self.api_key:
            self.api_key = self.gemini_api_key or self.google_api_key
        if not self.gemini_api_key:
            self.gemini_api_key = self.api_key
        if not self.google_api_key:
            self.google_api_key = self.api_key

# Load settings from .env file (fails fast if keys are missing)
settings = Settings()
