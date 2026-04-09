"""
Central config — reads from environment variables.
All other files import from here instead of calling os.getenv directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "mistral")
MODEL_DIR       = os.getenv("MODEL_DIR",       os.path.join(os.path.dirname(__file__), "saved"))