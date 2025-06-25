#!/usr/bin/env python3
"""
test_lexical_search
===================

Test suite for validating the lexical search functionality using BM25Retriever.

This script loads a precomputed BM25 index, performs a search query, and
displays the ranked results in detail.
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

from core.retrieval_engine.lexical_search import BM25Retriever


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Return the configuration dictionary loaded from the given YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_separator(title: str, char: str = "=") -> None:
    """Display a formatted separator with a title.

    Args:
        title (str): Section label.
        char (str): Separator character.
    """
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_result(result: Dict[str, Any], index: int) -> None:
    """Format and display one search result.

    Args:
        result (dict): The result dictionary containing 'document_id', 'chunk_text', and 'score'.
        index (int): Position of the result in the ranked list.
    """
    print(f"\nResult #{index + 1}")
    print(f"Document ID: {result['document_id']}")
    print(f"Chunk Text: {result['chunk_text']}")
    print(f"Score     : {result['score']:.4f}")


# --------------------------------------------------------------------------- #
# Main test procedure                                                         #
# --------------------------------------------------------------------------- #

def run_test_bm25() -> None:
    """Run a test query on the BM25 lexical retriever and display the results."""

    print_separator("Lexical Search with BM25")

    settings = load_settings()
    index_dir = settings["bm25"]["index_dir"]
    retriever = BM25Retriever(index_dir=index_dir)

    query = "turbine belt broken"
    print(f"Query: {query}\n")

    results = retriever.search(query, top_k=5)

    if not results:
        print("No results found.")
    else:
        for i, res in enumerate(results):
            print_result(res, i)


if __name__ == "__main__":
    run_test_bm25()
