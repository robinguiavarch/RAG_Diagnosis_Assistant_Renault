#!/usr/bin/env python3
"""
test_hybrid_fusion
==================

Test suite for validating the hybrid retrieval fusion system combining
BM25 and FAISS results. It checks score normalization and the fusion
logic between the lexical and semantic retrievers.

Run the test with controlled threading::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_hybrid_fusion.py \
        > tests/results/test_fusion.txt 2>&1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# --------------------------------------------------------------------------- #
# Project path setup                                                          #
# --------------------------------------------------------------------------- #

# Add the project root to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.retrieval_engine.hybrid_fusion import normalize_scores, fuse_results
from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Return the configuration dictionary loaded from the given YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_separator(title: str, char: str = "=") -> None:
    """Display a separator line with a centered title.

    Args:
        title (str): The label to display within the separator.
        char (str): The character to repeat around the label.
    """
    print(f"\n{char * 10} {title.upper()} {char * 10}\n")


# --------------------------------------------------------------------------- #
# Main test procedure                                                         #
# --------------------------------------------------------------------------- #

def run_test_hybrid_fusion() -> None:
    """Run end-to-end test for hybrid fusion between BM25 and FAISS retrievers.

    This function instantiates the lexical and semantic retrievers, queries
    both with a test string, normalizes their scores, and fuses the ranked
    results using a uniform weighting scheme (0.5, 0.5).
    """

    print_separator("Hybrid Fusion Test")

    settings = load_settings()

    # ----------------------------------------------------------------------- #
    # Instantiate retrievers                                                 #
    # ----------------------------------------------------------------------- #
    bm25 = BM25Retriever(index_dir=settings["bm25"]["index_dir"])
    faiss = FAISSRetriever(
        embedding_dir=settings["semantic"]["embedding_dir"],
        index_path=settings["semantic"]["index_path"],
        normalize_embeddings=True,
    )

    query = "belt broken on turbine"

    print(f"Query: {query}\n")

    results_bm25 = bm25.search(query, top_k=5)
    results_faiss = faiss.search(query, top_k=5)

    print_separator("BM25 Results")
    for rank, doc in enumerate(results_bm25, 1):
        print(f"{rank}. {doc['id']} | score={doc['score']:.4f}")

    print_separator("FAISS Results")
    for rank, doc in enumerate(results_faiss, 1):
        print(f"{rank}. {doc['id']} | score={doc['score']:.4f}")

    # ----------------------------------------------------------------------- #
    # Normalize and fuse                                                     #
    # ----------------------------------------------------------------------- #

    norm_bm25 = normalize_scores(results_bm25)
    norm_faiss = normalize_scores(results_faiss)

    fused = fuse_results(norm_bm25, norm_faiss, weight_bm25=0.5, weight_faiss=0.5)

    print_separator("Fused Results")
    for rank, doc in enumerate(fused, 1):
        print(f"{rank}. {doc['id']} | fused_score={doc['score']:.4f}")


if __name__ == "__main__":
    run_test_hybrid_fusion()
