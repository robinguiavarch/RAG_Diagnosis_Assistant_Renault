"""
Test simple des 4 systèmes avec cloud
Path: tests/test_cloud_simple.py
"""

import os
import sys
sys.path.append('..')  # Accès au dossier racine depuis tests/

from dotenv import load_dotenv
load_dotenv()

def test_queriers():
    """Test des 3 queriers avec cloud"""
    try:
        from core.retrieval_graph.dense_kg_querier import get_structured_context
        from core.retrieval_graph.sparse_kg_querier import get_structured_context_sparse  
        from core.retrieval_graph.dense_sc_kg_querier import get_structured_context_dense_sc
        
        query = "motor overheating FANUC"
        
        # Test Dense
        result_dense = get_structured_context(query, max_triplets=2)
        print(f"Dense: {'✅' if 'Triplet' in result_dense else '❌'}")
        
        # Test Sparse  
        result_sparse = get_structured_context_sparse(query, max_triplets=2)
        print(f"Sparse: {'✅' if 'Triplet' in result_sparse else '❌'}")
        
        # Test Dense S&C
        result_sc = get_structured_context_dense_sc(query, max_triplets=2)
        print(f"Dense S&C: {'✅' if 'Triplet' in result_sc else '❌'}")
        
    except Exception as e:
        print(f"❌ {e}")

def test_streamlit_import():
    """Test import composants Streamlit"""
    try:
        from core.retrieval_engine.lexical_search import BM25Retriever
        from core.retrieval_engine.semantic_search import FAISSRetriever
        from core.response_generation.standard_rag_generator import OpenAIGenerator
        print("Streamlit imports: ✅")
    except Exception as e:
        print(f"Streamlit imports: ❌ {e}")

if __name__ == "__main__":
    print("🧪 Test cloud simple")
    test_queriers()
    test_streamlit_import()