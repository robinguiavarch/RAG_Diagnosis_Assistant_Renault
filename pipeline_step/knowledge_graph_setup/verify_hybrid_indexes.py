"""
Vérification des index BM25 et FAISS pour métrique hybride
Path: pipeline_step/knowledge_graph_setup/verify_hybrid_indexes.py
"""

import os
from pathlib import Path

def verify_dense_indexes():
    """Vérifie les index Dense (symptômes)"""
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_dense"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_dense"
    
    print("📊 Vérification index Dense...")
    print(f"BM25: {'✅' if bm25_path.exists() else '❌'} {bm25_path}")
    print(f"FAISS: {'✅' if faiss_path.exists() else '❌'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_sparse_indexes():
    """Vérifie les index Sparse (symptômes)"""
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_sparse"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_sparse"
    
    print("📊 Vérification index Sparse...")
    print(f"BM25: {'✅' if bm25_path.exists() else '❌'} {bm25_path}")
    print(f"FAISS: {'✅' if faiss_path.exists() else '❌'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_dense_sc_indexes():
    """Vérifie les index Dense S&C (symptômes + causes)"""
    script_dir = Path(__file__).parent.parent.parent
    
    bm25_path = script_dir / "data" / "indexes" / "symptom_bm25_dense_sc"
    faiss_path = script_dir / "data" / "knowledge_base" / "symptom_embeddings_dense_s&c"
    
    print("📊 Vérification index Dense S&C...")
    print(f"BM25: {'✅' if bm25_path.exists() else '❌'} {bm25_path}")
    print(f"FAISS: {'✅' if faiss_path.exists() else '❌'} {faiss_path}")
    
    return bm25_path.exists() and faiss_path.exists()

def verify_all_indexes():
    """Vérifie tous les index pour métrique hybride"""
    print("🔍 Vérification des index pour métrique hybride KG")
    print("=" * 60)
    
    dense_ok = verify_dense_indexes()
    print()
    
    sparse_ok = verify_sparse_indexes()
    print()
    
    dense_sc_ok = verify_dense_sc_indexes()
    print()
    
    print("📋 RÉSUMÉ:")
    print(f"Dense: {'✅' if dense_ok else '❌ MANQUANT'}")
    print(f"Sparse: {'✅' if sparse_ok else '❌ MANQUANT'}")
    print(f"Dense S&C: {'✅' if dense_sc_ok else '❌ MANQUANT'}")
    
    if not (dense_ok and sparse_ok and dense_sc_ok):
        print("\n⚠️ Index manquants détectés!")
        print("Exécutez les scripts de construction d'index avant la métrique hybride:")
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
        print("\n✅ Tous les index sont présents!")
        print("Métrique hybride prête pour construction KG")

if __name__ == "__main__":
    verify_all_indexes()