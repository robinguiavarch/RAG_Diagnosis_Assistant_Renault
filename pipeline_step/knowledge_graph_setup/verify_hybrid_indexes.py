"""
Hybrid Index Verification Module for Knowledge Graph Construction

This module provides verification capabilities for BM25 and FAISS indexes required
for the hybrid metric Knowledge Graph construction pipeline. It validates the presence
and accessibility of all necessary index components across Dense, Sparse, and Dense S&C
Knowledge Graph configurations.

Key components:
- Index verification: Systematic validation of BM25 and FAISS index availability
- Multi-KG support: Verification for Dense, Sparse, and Dense S&C configurations
- Diagnostic guidance: Clear reporting and actionable recommendations for missing components

Dependencies: pathlib, os
Usage: Executed before hybrid metric KG construction to ensure all prerequisites are met
Path: pipeline_step/knowledge_graph_setup/verify_hybrid_indexes.py
"""

import os
from pathlib import Path

def verify_dense_indexes():
    """
    Verify Dense Knowledge Graph indexes for hybrid metric computation
    
    Checks for the presence of both BM25 lexical and FAISS semantic indexes
    required for Dense KG symptom similarity calculations in the hybrid metric.
    
    Returns:
        bool: True if both BM25 and FAISS indexes are present, False otherwise
    """
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_dense"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_dense"
    
    print("Dense index verification...")
    print(f"BM25: {'PRESENT' if bm25_path.exists() else 'MISSING'} {bm25_path}")
    print(f"FAISS: {'PRESENT' if faiss_path.exists() else 'MISSING'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_sparse_indexes():
    """
    Verify Sparse Knowledge Graph indexes for hybrid metric computation
    
    Checks for the presence of both BM25 lexical and FAISS semantic indexes
    required for Sparse KG symptom similarity calculations in the hybrid metric.
    
    Returns:
        bool: True if both BM25 and FAISS indexes are present, False otherwise
    """
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_sparse"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_sparse"
    
    print("Sparse index verification...")
    print(f"BM25: {'PRESENT' if bm25_path.exists() else 'MISSING'} {bm25_path}")
    print(f"FAISS: {'PRESENT' if faiss_path.exists() else 'MISSING'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_dense_sc_indexes():
    """
    Verify Dense Symptom & Cause Knowledge Graph indexes for hybrid metric computation
    
    Checks for the presence of both BM25 lexical and FAISS semantic indexes
    required for Dense S&C KG combined text similarity calculations in the hybrid metric.
    
    Returns:
        bool: True if both BM25 and FAISS indexes are present, False otherwise
    """
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_dense_sc"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_dense_s&c"
    
    print("Dense S&C index verification...")
    print(f"BM25: {'PRESENT' if bm25_path.exists() else 'MISSING'} {bm25_path}")
    print(f"FAISS: {'PRESENT' if faiss_path.exists() else 'MISSING'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_all_indexes():
    """
    Comprehensive verification of all indexes required for hybrid metric KG construction
    
    Performs systematic verification of all BM25 and FAISS indexes across the three
    Knowledge Graph configurations (Dense, Sparse, Dense S&C). Provides detailed
    reporting and actionable guidance for resolving any missing components.
    
    The function ensures that all prerequisite indexes are available before
    attempting hybrid metric Knowledge Graph construction, preventing runtime
    errors and providing clear diagnostic information.
    """
    print("Hybrid metric KG index verification")
    print("=" * 60)
    
    dense_ok = verify_dense_indexes()
    print()
    
    sparse_ok = verify_sparse_indexes()
    print()
    
    dense_sc_ok = verify_dense_sc_indexes()
    print()
    
    print("VERIFICATION SUMMARY:")
    print(f"Dense: {'COMPLETE' if dense_ok else 'MISSING COMPONENTS'}")
    print(f"Sparse: {'COMPLETE' if sparse_ok else 'MISSING COMPONENTS'}")
    print(f"Dense S&C: {'COMPLETE' if dense_sc_ok else 'MISSING COMPONENTS'}")
    
    if not (dense_ok and sparse_ok and dense_sc_ok):
        print("\nMissing indexes detected")
        print("Execute the following index construction scripts before running hybrid metric:")
        if not dense_ok:
            print("  - build_symptom_bm25_index_kg_dense.py")
            print("  - build_symptom_vector_index_kg_dense.py")
        if not sparse_ok:
            print("  - build_symptom_bm25_index_kg_sparse.py")
            print("  - build_symptom_vector_index_kg_sparse.py")
        if not dense_sc_ok:
            print("  - build_symptom_bm25_index_kg_dense_s&c.py")
            print("  - build_symptom_vector_index_kg_dense_s&c.py")
    else:
        print("\nAll indexes are present and accessible")
        print("Hybrid metric system ready for Knowledge Graph construction")

if __name__ == "__main__":
    verify_all_indexes()