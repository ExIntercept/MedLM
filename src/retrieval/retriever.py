"""Hybrid retrieval engine: BM25 (sparse) + ChromaDB (dense) + RRF + MedCPT reranking."""
import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import chromadb
import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import CHROMA_DB_DIR, CORPUS_DIR

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

SOURCE_WEIGHTS = {"StatPearls": 1.3, "OpenFDA": 1.1, "MedQuAD": 0.8}
MEDICATION_BM25_REPEAT = 4
MEDICATION_TITLE_BOOST = 1.6


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
print(f"🔧 Using compute device for RAG: {DEVICE}")

# 1. Load BM25 Index & Corpus Chunks
print("📦 [1/3] Loading BM25 Index & Corpus Chunks...")
bm25_path = CORPUS_DIR / "bm25_index.pkl"
chunks_path = CORPUS_DIR / "full_corpus_chunks.csv"

if not bm25_path.exists() or not chunks_path.exists():
    raise FileNotFoundError(f"Missing corpus assets in {CORPUS_DIR}. Expected bm25_index.pkl and full_corpus_chunks.csv")

with open(bm25_path, "rb") as f:
    bm25 = pickle.load(f)["bm25"]
chunks_df = pd.read_csv(chunks_path)

KNOWN_MEDICATIONS = set(
    chunks_df.loc[chunks_df["source"] == "OpenFDA", "title"].dropna().str.lower().str.strip().unique()
)

# 2. ChromaDB Client & Lazy Embedder
print("📦 [2/3] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
try:
    collection = chroma_client.get_or_create_collection(name="medical_corpus")
    coll_count = collection.count()
except Exception as e:
    print(f"Warning: ChromaDB collection initialization note: {e}")
    collection = None
    coll_count = 0

print(f"   Loading S-PubMedBERT Embedder on {DEVICE}...")
try:
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device=DEVICE)
except Exception as exc:
    print(f"Warning loading embedder on {DEVICE}, falling back to CPU: {exc}")
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device="cpu")


def get_embedder() -> SentenceTransformer:
    return embedder


# 3. Load MedCPT Neural Cross-Encoder Reranker
print(f"📦 [3/3] Loading MedCPT Neural Cross-Encoder Reranker on {DEVICE}...")
reranker = CrossEncoder("ncbi/MedCPT-Cross-Encoder", device=DEVICE, trust_remote_code=True)
print("✅ Hybrid RAG Pipeline Ready.")


def biomedical_tokenize(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return [stemmer.stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]


_HAS_DIGIT = re.compile(r"\d")


def extract_intent(vignette_text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", vignette_text.strip()) if s.strip()]
    if len(sentences) <= 6:
        return vignette_text
    presentation = sentences[0]
    lead_in = sentences[-1]
    findings = [
        s
        for s in sentences[1:-1]
        if any(
            k in s.lower()
            for k in [
                "shows", "reveals", "diagnosed", "treatment", "elevated",
                "decreased", "positive", "mass", "pain", "history of",
            ]
        )
        or _HAS_DIGIT.search(s)
        or any(med in s.lower() for med in KNOWN_MEDICATIONS)
    ]
    # Query expansion for colloquial / ambiguous clinical terms
    expanded = intent_query
    if re.search(r"\bbicuspid\b", intent_query, re.I) and "aortic" not in intent_query.lower():
        expanded += " bicuspid aortic valve aortic stenosis aortopathy"
    if re.search(r"\bheart\s*pain\b", intent_query, re.I):
        expanded += " cardiac chest pain angina myocardial"
    return expanded.strip()


def extract_medication_entities(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    seen = []
    for w in words:
        if w in KNOWN_MEDICATIONS and w not in seen:
            seen.append(w)
    return seen


def retrieve_context(vignette_text: str, top_k: int = 3) -> List[Tuple[str, str, Dict[str, Any]]]:
    intent_query = extract_intent(vignette_text)
    medication_entities = extract_medication_entities(intent_query)

    def title_matches_entity(meta):
        if not medication_entities:
            return False
        title = str(meta.get("title", "")).lower()
        return any(entity in title for entity in medication_entities)

    # 1. Sparse Lexical Retrieval via BM25 (always available across all 30,980 chunks)
    tokens = biomedical_tokenize(intent_query)
    for entity in medication_entities:
        tokens.extend(biomedical_tokenize(entity) * MEDICATION_BM25_REPEAT)
    sparse_scores = bm25.get_scores(tokens)
    top_sparse_idx = sparse_scores.argsort()[::-1][:30]
    sparse_tuples = [
        (
            chunks_df.iloc[i]["chunk_id"],
            chunks_df.iloc[i]["text"],
            chunks_df.iloc[i][["source", "title", "field"]].to_dict(),
        )
        for i in top_sparse_idx
    ]

    # 2. Dense Vector Retrieval via ChromaDB (if indexed)
    dense_tuples = []
    if collection is not None and collection.count() > 0:
        try:
            emb = get_embedder()
            query_emb = emb.encode([intent_query]).tolist()
            dense_res = collection.query(query_embeddings=query_emb, n_results=min(25, collection.count()))
            if dense_res and dense_res.get("ids") and len(dense_res["ids"][0]) > 0:
                dense_tuples = list(zip(dense_res["ids"][0], dense_res["documents"][0], dense_res["metadatas"][0]))
        except Exception as exc:
            print(f"Warning: Dense ChromaDB query error: {exc}")

    # 3. RRF Fusion
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for rank, (cid, text, meta) in enumerate(dense_tuples):
        w = SOURCE_WEIGHTS.get(meta.get("source"), 1.0)
        if title_matches_entity(meta):
            w *= MEDICATION_TITLE_BOOST
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (60 + rank + 1)) * w
        doc_lookup[cid] = (text, meta)

    for rank, (cid, text, meta) in enumerate(sparse_tuples):
        w = SOURCE_WEIGHTS.get(meta.get("source"), 1.0)
        if title_matches_entity(meta):
            w *= MEDICATION_TITLE_BOOST
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (60 + rank + 1)) * w
        doc_lookup[cid] = (text, meta)

    candidates = [
        (cid, doc_lookup[cid][0], doc_lookup[cid][1])
        for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:35]
    ]
    if not candidates:
        return []

    # 4. Neural Cross-Encoder Reranking via MedCPT
    pairs = [(intent_query, text) for _, text, _ in candidates]
    c_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    scored = sorted(zip(candidates, c_scores), key=lambda x: x[1], reverse=True)

    deduped = []
    seen = set()
    for (cid, text, meta), score in scored:
        if cid not in seen:
            seen.add(cid)
            meta_copy = dict(meta)
            meta_copy["cross_score"] = float(score)
            deduped.append((cid, text, meta_copy))
        if len(deduped) >= top_k:
            break
    return deduped
