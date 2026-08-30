"""Main server entry point for Medical RAG & MedGemma."""
import uvicorn
from src.api.server import app

if __name__ == "__main__":
    print("=" * 65)
    print("🏥 LAUNCHING MEDICAL RAG SYSTEM (MedGemma on Colab via ngrok)")
    print("=" * 65)
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8600, reload=True)
