"""Offline/Background ChromaDB dense indexing script for medical corpus."""
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from src.config import CHROMA_DB_DIR, CORPUS_DIR
from src.retrieval.retriever import get_device

def main():
    device = get_device()
    print(f"🔧 Device for indexing: {device}")
    chunks_path = CORPUS_DIR / "full_corpus_chunks.csv"
    chunks_df = pd.read_csv(chunks_path)
    print(f"📦 Loaded {len(chunks_df)} chunks from {chunks_path}")

    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=device)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_or_create_collection(name="medical_corpus")

    existing_count = collection.count()
    print(f"Current ChromaDB collection count: {existing_count}")
    if existing_count >= len(chunks_df):
        print("✅ ChromaDB is already fully indexed.")
        return

    batch_size = 256
    total_chunks = len(chunks_df)
    t0 = time.time()

    for i in range(existing_count, total_chunks, batch_size):
        batch = chunks_df.iloc[i : i + batch_size]
        ids = batch["chunk_id"].astype(str).tolist()
        texts = batch["text"].fillna("").astype(str).tolist()
        metas = batch[["source", "title", "field"]].fillna("").to_dict(orient="records")
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
        collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
        pct = round((min(i + batch_size, total_chunks) / total_chunks) * 100, 1)
        print(f"[{pct}%] Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

    elapsed = round(time.time() - t0, 1)
    print(f"✅ ChromaDB indexing complete ({collection.count()} chunks) in {elapsed}s.")

if __name__ == "__main__":
    main()
