"""
Hybrid Fusion: Advanced Search Result Combination and Deduplication

This module provides sophisticated fusion capabilities for combining lexical and semantic
search results in the RAG diagnosis system. It implements score normalization, weighted
fusion algorithms, and intelligent deduplication mechanisms to optimize hybrid search
performance and result quality.

Key components:
- Min-max score normalization for consistent cross-retriever comparison
- Weighted fusion algorithm combining BM25 lexical and FAISS semantic results
- Intelligent deduplication using content-based hashing
- Source tracking and mixed result identification
- Configurable weighting parameters for lexical vs semantic balance

Dependencies: numpy, collections, hashlib
Usage: Import fusion functions for combining and deduplicating hybrid search results
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import hashlib


def normalize_scores(results: List[Dict], key: str = "score") -> List[Dict]:
    """
    Normalize retriever scores between 0 and 1 using min-max scaling.
    
    Applies min-max normalization to ensure consistent score ranges across
    different retrieval methods, adding a 'normalized_score' field to each document.
    
    Args:
        results (List[Dict]): List of search results with scores
        key (str): Score field name to normalize
        
    Returns:
        List[Dict]: Results with added normalized_score field
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
    Fuse BM25 and FAISS results using weighted average of normalized scores.
    
    Combines lexical and semantic search results through score normalization
    and weighted fusion, with source tracking for result provenance analysis.
    
    Args:
        bm25_results (List[Dict]): BM25 lexical search results
        faiss_results (List[Dict]): FAISS semantic search results
        top_k (int): Number of top results to return
        alpha (float): Weight for lexical (BM25) vs semantic (FAISS) scores
        
    Returns:
        List[Dict]: Fused results sorted by combined score with source attribution
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
            fused_dict[key]["source"] = "Mixed"
        else:
            fused_dict[key]["source"] = "Semantic (FAISS)"

    # Final score: weighted average
    for doc in fused_dict.values():
        doc["fused_score"] = alpha * doc["bm25_score"] + (1 - alpha) * doc["faiss_score"]

    # Sort by fused score in descending order
    fused_list = list(fused_dict.values())
    fused_list.sort(key=lambda d: d["fused_score"], reverse=True)

    return fused_list[:top_k]

def deduplicate_by_content_hash(candidates: List[Dict]) -> List[Dict]:
    """
    Intelligent deduplication based on content hashing.
    
    Removes duplicate documents by computing MD5 hashes of content text,
    preserving only the first occurrence of each unique content piece.
    Used by the enhanced retrieval engine for result optimization.
    
    Args:
        candidates (List[Dict]): List of candidate documents to deduplicate
        
    Returns:
        List[Dict]: Deduplicated list with unique content only
    """
    if not candidates:
        return []
    
    unique_docs = []
    seen_hashes = set()
    
    for doc in candidates:
        content = doc.get("text", "").strip()
        if not content:
            continue
        
        # Content hash to detect exact duplicates
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs