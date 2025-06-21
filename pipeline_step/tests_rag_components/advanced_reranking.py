import yaml
import requests
import os
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever
from core.retrieval_engine.hybrid_fusion import fuse_results

# Chargement environnement et configuration
load_dotenv()

def load_settings(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

settings = load_settings()
paths = settings["paths"]
rerank_cfg = settings["reranking"]

# Chargement retrievers
bm25 = BM25Retriever(index_dir=paths["bm25_index"])
faiss = FAISSRetriever(
    index_path=paths["faiss_index"],
    metadata_path=paths["embedding_file"]
)

# HuggingFace CrossEncoder API
HF_API_KEY = os.getenv("API_TOKEN_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def rerank_via_api(query, passages, top_k=5):
    payload = {
        "inputs": {
            "source_sentence": query,
            "sentences": passages
        }
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    scores = response.json()
    scored_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, s in scored_passages[:top_k]]

# Pipeline principal pour tests rapides (CLI)
def main():
    query = input("💬 Entrez votre requête : ")
    
    bm25_results = bm25.search(query, top_k=rerank_cfg.get("top_k_before_rerank", 10))
    faiss_results = faiss.search(query, top_k=rerank_cfg.get("top_k_before_rerank", 10))
    fused = fuse_results(bm25_results, faiss_results, top_k=rerank_cfg.get("top_k_before_rerank", 10))
    fused_passages = [doc["text"] for doc in fused]

    reranked_passages = rerank_via_api(query, fused_passages, top_k=rerank_cfg.get("final_top_k", 3))

    print("\n📄 Passages après reranking :")
    for i, p in enumerate(reranked_passages, 1):
        print(f"\n--- Passage {i} ---\n{p}")

if __name__ == "__main__":
    main()
