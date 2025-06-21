# src/retrieval/fusion.py

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


def normalize_scores(results: List[Dict], key: str = "score") -> List[Dict]:
    """
    Normalise les scores d'un retriever entre 0 et 1 (min-max scaling).
    Ajoute un champ 'normalized_score' à chaque document.
    """
    if not results:
        return results

    scores = np.array([doc.get(key, 0.0) for doc in results])
    min_score = scores.min()
    max_score = scores.max()
    denominator = max_score - min_score if max_score != min_score else 1.0

    for doc in results:
        raw_score = doc.get(key, 0.0)
        doc["normalized_score"] = (raw_score - min_score) / denominator
    return results


def fuse_results(
    bm25_results: List[Dict],
    faiss_results: List[Dict],
    top_k: int = 5,
    alpha: float = 0.5
) -> List[Dict]:
    """
    Fusionne les résultats BM25 et FAISS via une moyenne pondérée des scores normalisés.

    - alpha : poids du lexical (BM25) vs sémantique (FAISS)
    - Chaque document reçoit un champ 'source' (BM25, FAISS ou Mixte)
    """
    bm25_results = normalize_scores(bm25_results, key="score")
    faiss_results = normalize_scores(faiss_results, key="score")

    fused_dict = defaultdict(lambda: {
        "text": "",
        "bm25_score": 0.0,
        "faiss_score": 0.0,
        "fused_score": 0.0,
        "source": ""
    })

    for doc in bm25_results:
        key = doc["text"]
        fused_dict[key]["text"] = key
        fused_dict[key]["bm25_score"] = doc["normalized_score"]
        fused_dict[key]["source"] = "Lexical (BM25)"

    for doc in faiss_results:
        key = doc["text"]
        fused_dict[key]["text"] = key
        fused_dict[key]["faiss_score"] = doc["normalized_score"]
        if fused_dict[key]["source"]:
            fused_dict[key]["source"] = "Mixte"
        else:
            fused_dict[key]["source"] = "Sémantique (FAISS)"

    # Score final : moyenne pondérée
    for doc in fused_dict.values():
        doc["fused_score"] = alpha * doc["bm25_score"] + (1 - alpha) * doc["faiss_score"]

    # Tri décroissant par score fusionné
    fused_list = list(fused_dict.values())
    fused_list.sort(key=lambda d: d["fused_score"], reverse=True)

    return fused_list[:top_k]

def deduplicate_by_content_hash(candidates: List[Dict]) -> List[Dict]:
    """
    Déduplication intelligente basée sur le hash du contenu
    Utilisée par le nouveau EnhancedRetrievalEngine
    
    Args:
        candidates: Liste des candidats à dédupliquer
        
    Returns:
        Liste déduplicquée
    """
    if not candidates:
        return []
    
    unique_docs = []
    seen_hashes = set()
    
    for doc in candidates:
        content = doc.get("text", "").strip()
        if not content:
            continue
        
        # Hash du contenu pour détecter les doublons exacts
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs