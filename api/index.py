"""Vercel entry point — re-exports the FastAPI app."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from main import app  # noqa: F401