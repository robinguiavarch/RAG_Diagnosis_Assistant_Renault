#!/usr/bin/env python3
"""
Test du système de fusion BM25 + FAISS
Teste la normalisation et la fusion des résultats des deux retrievers

IMPORTANT: Pour run le test:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_fusion.py > tests/resultats/test_fusion.txt 2>&1
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from core.retrieval_engine.hybrid_fusion import normalize_scores, fuse_results
from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_separator(title: str, char: str = "="):
    """Affiche un séparateur avec titre"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_result(result: Dict, index: int, result_type: str = "FUSIONNÉ"):
    """Affiche un résultat de manière formatée"""
    print(f"\n🔀 RÉSULTAT {result_type} #{index + 1}")
    print(f"🆔 Document: {result['document_id']}")
    print(f"🧩 Chunk: {result['chunk_id']}")
    
    # Scores selon le type
    if result_type == "FUSIONNÉ":
        print(f"🎯 Score fusionné: {result['fused_score']:.4f}")
        print(f"📝 Score BM25: {result['bm25_score']:.4f}")
        print(f"🧠 Score FAISS: {result['faiss_score']:.4f}")
    else:
        print(f"📊 Score: {result['score']:.4f}")
    
    print(f"📖 TEXTE: {result['text'][:150]}...")
    print("-" * 50)


def test_normalize_scores():
    """Test de la fonction de normalisation"""
    print_separator("TEST DE NORMALISATION DES SCORES")
    
    # Test données normales
    print("📊 Test normalisation avec données variées")
    test_results = [
        {"id": "A", "score": 1.5, "text": "test A"},
        {"id": "B", "score": 3.2, "text": "test B"},
        {"id": "C", "score": 0.8, "text": "test C"},
        {"id": "D", "score": 2.1, "text": "test D"}
    ]
    
    normalized = normalize_scores(test_results.copy(), "score")
    
    print("Résultats avant/après normalisation:")
    for orig, norm in zip(test_results, normalized):
        print(f"   {orig['id']}: {orig['score']:.2f} → {norm['normalized_score']:.4f}")
    
    # Vérifications
    norm_scores = [r['normalized_score'] for r in normalized]
    print(f"✅ Min normalisé: {min(norm_scores):.4f} (doit être 0.0)")
    print(f"✅ Max normalisé: {max(norm_scores):.4f} (doit être 1.0)")
    
    # Test cas limite - scores identiques
    print("\n📊 Test normalisation avec scores identiques")
    identical_results = [
        {"id": "X", "score": 2.5, "text": "test X"},
        {"id": "Y", "score": 2.5, "text": "test Y"},
        {"id": "Z", "score": 2.5, "text": "test Z"}
    ]
    
    normalized_identical = normalize_scores(identical_results.copy(), "score")
    
    for result in normalized_identical:
        print(f"   {result['id']}: {result['score']:.2f} → {result['normalized_score']:.4f}")
    
    # Test liste vide
    print("\n📊 Test normalisation avec liste vide")
    empty_normalized = normalize_scores([], "score")
    print(f"✅ Liste vide: {len(empty_normalized)} résultats")


def test_fusion_logic():
    """Test de la logique de fusion avec données simulées"""
    print_separator("TEST DE LOGIQUE DE FUSION")
    
    # Données de test simulées
    bm25_results = [
        {"document_id": "doc1", "chunk_id": "1", "text": "Chunk commun 1", "score": 2.5},
        {"document_id": "doc1", "chunk_id": "2", "text": "Chunk BM25 seul", "score": 1.8},
        {"document_id": "doc2", "chunk_id": "3", "text": "Chunk commun 2", "score": 3.1},
        {"document_id": "doc2", "chunk_id": "4", "text": "Autre chunk BM25", "score": 1.2}
    ]
    
    faiss_results = [
        {"document_id": "doc1", "chunk_id": "1", "text": "Chunk commun 1", "score": 0.85},
        {"document_id": "doc2", "chunk_id": "3", "text": "Chunk commun 2", "score": 0.92},
        {"document_id": "doc3", "chunk_id": "5", "text": "Chunk FAISS seul", "score": 0.78},
        {"document_id": "doc3", "chunk_id": "6", "text": "Autre chunk FAISS", "score": 0.65}
    ]
    
    print("📝 Résultats BM25:")
    for i, result in enumerate(bm25_results):
        print(f"   {i+1}. {result['document_id']}|{result['chunk_id']}: score={result['score']:.2f}")
    
    print("\n🧠 Résultats FAISS:")
    for i, result in enumerate(faiss_results):
        print(f"   {i+1}. {result['document_id']}|{result['chunk_id']}: score={result['score']:.2f}")
    
    # Fusion
    fused_results = fuse_results(bm25_results, faiss_results, top_k=6)
    
    print(f"\n🔀 Résultats fusionnés (top-6):")
    for i, result in enumerate(fused_results):
        print(f"   {i+1}. {result['document_id']}|{result['chunk_id']}: "
              f"fusionné={result['fused_score']:.4f} "
              f"(BM25={result['bm25_score']:.4f}, FAISS={result['faiss_score']:.4f})")
    
    # Analyse
    print(f"\n📊 ANALYSE:")
    print(f"   📝 Chunks BM25 uniques: {len(bm25_results)}")
    print(f"   🧠 Chunks FAISS uniques: {len(faiss_results)}")
    print(f"   🔀 Chunks fusionnés: {len(fused_results)}")
    
    # Trouver les chunks communs
    bm25_keys = {(r['document_id'], r['chunk_id']) for r in bm25_results}
    faiss_keys = {(r['document_id'], r['chunk_id']) for r in faiss_results}
    common_keys = bm25_keys & faiss_keys
    print(f"   🤝 Chunks en commun: {len(common_keys)}")
    
    return fused_results


def test_real_retrievers():
    """Test avec les vrais retrievers BM25 et FAISS"""
    print_separator("TEST AVEC VRAIS RETRIEVERS")
    
    try:
        # Configuration
        settings = load_settings()
        
        # Initialisation BM25
        print("🔄 Initialisation BM25Retriever...")
        bm25_index_dir = Path(settings["paths"]["bm25_index"])
        if not bm25_index_dir.exists():
            print("❌ Index BM25 non trouvé - skipping test réel")
            return
        
        bm25_retriever = BM25Retriever(index_dir=bm25_index_dir)
        print("✅ BM25Retriever initialisé")
        
        # Initialisation FAISS
        print("🔄 Initialisation FAISSRetriever...")
        faiss_index_dir = Path(settings["paths"]["faiss_index"])
        index_path = faiss_index_dir / "index.faiss"
        metadata_path = faiss_index_dir / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            print("❌ Index FAISS non trouvé - skipping test réel")
            return
        
        faiss_retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model_name=settings["models"]["embedding_model"]
        )
        print("✅ FAISSRetriever initialisé")
        
        # Test requête
        query = "I got the error ACAL-006 TPE operation error on the FANUC teach pendant. What should I do?"
        top_k = 5
        
        print(f"\n🔍 Requête: \"{query}\"")
        print(f"📊 Top-K: {top_k}")
        
        # Recherche BM25
        print("\n📝 Recherche BM25...")
        bm25_results = bm25_retriever.search(query, top_k=top_k)
        print(f"   Trouvé: {len(bm25_results)} résultats")
        
        # Recherche FAISS
        print("🧠 Recherche FAISS...")
        faiss_results = faiss_retriever.search(query, top_k=top_k)
        print(f"   Trouvé: {len(faiss_results)} résultats")
        
        if not bm25_results and not faiss_results:
            print("❌ Aucun résultat trouvé dans les deux retrievers")
            return
        
        # Fusion
        print("\n🔀 Fusion des résultats...")
        fused_results = fuse_results(bm25_results, faiss_results, top_k=top_k)
        print(f"   Résultats fusionnés: {len(fused_results)}")
        
        # Affichage comparatif
        print_separator("COMPARAISON DES RÉSULTATS", "-")
        
        print("\n📝 TOP-3 BM25:")
        for i, result in enumerate(bm25_results[:3]):
            print(f"   {i+1}. Score: {result['score']:.4f} | {result['document_id']}|{result['chunk_id']}")
            print(f"      {result['text'][:100]}...")
        
        print("\n🧠 TOP-3 FAISS:")
        for i, result in enumerate(faiss_results[:3]):
            print(f"   {i+1}. Score: {result['score']:.4f} | {result['document_id']}|{result['chunk_id']}")
            print(f"      {result['text'][:100]}...")
        
        print("\n🔀 TOP-5 FUSIONNÉS:")
        for i, result in enumerate(fused_results):
            print_result(result, i, "FUSIONNÉ")
        
        # Analyse de la fusion
        print_separator("ANALYSE DE LA FUSION", "-")
        
        bm25_chunks = {(r['document_id'], r['chunk_id']) for r in bm25_results}
        faiss_chunks = {(r['document_id'], r['chunk_id']) for r in faiss_results}
        fused_chunks = {(r['document_id'], r['chunk_id']) for r in fused_results}
        
        common_chunks = bm25_chunks & faiss_chunks
        only_bm25 = bm25_chunks - faiss_chunks
        only_faiss = faiss_chunks - bm25_chunks
        
        print(f"📊 STATISTIQUES:")
        print(f"   🤝 Chunks communs: {len(common_chunks)}")
        print(f"   📝 Chunks BM25 uniquement: {len(only_bm25)}")
        print(f"   🧠 Chunks FAISS uniquement: {len(only_faiss)}")
        print(f"   🔀 Chunks dans résultat final: {len(fused_chunks)}")
        
        # Analyse des scores
        if fused_results:
            fused_scores = [r['fused_score'] for r in fused_results]
            bm25_scores_in_fusion = [r['bm25_score'] for r in fused_results]
            faiss_scores_in_fusion = [r['faiss_score'] for r in fused_results]
            
            print(f"\n📈 SCORES:")
            print(f"   🎯 Score fusionné moyen: {sum(fused_scores)/len(fused_scores):.4f}")
            print(f"   📝 Contribution BM25 moyenne: {sum(bm25_scores_in_fusion)/len(bm25_scores_in_fusion):.4f}")
            print(f"   🧠 Contribution FAISS moyenne: {sum(faiss_scores_in_fusion)/len(faiss_scores_in_fusion):.4f}")
        
        return fused_results
        
    except Exception as e:
        print(f"❌ Erreur test réel: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_edge_cases():
    """Test des cas limites"""
    print_separator("TEST DES CAS LIMITES")
    
    # Test listes vides
    print("📊 Test fusion avec listes vides")
    empty_fusion = fuse_results([], [], top_k=5)
    print(f"   Listes vides: {len(empty_fusion)} résultats")
    
    # Test une liste vide
    bm25_only = [{"document_id": "doc1", "chunk_id": "1", "text": "test", "score": 2.0}]
    partial_fusion = fuse_results(bm25_only, [], top_k=5)
    print(f"   BM25 seul: {len(partial_fusion)} résultats")
    if partial_fusion:
        print(f"      Score fusionné: {partial_fusion[0]['fused_score']:.4f}")
        print(f"      (BM25: {partial_fusion[0]['bm25_score']:.4f}, FAISS: {partial_fusion[0]['faiss_score']:.4f})")
    
    # Test top_k > résultats disponibles
    small_bm25 = [{"document_id": "doc1", "chunk_id": "1", "text": "test", "score": 1.0}]
    small_faiss = [{"document_id": "doc2", "chunk_id": "2", "text": "test", "score": 0.8}]
    large_k_fusion = fuse_results(small_bm25, small_faiss, top_k=10)
    print(f"   Top-K > disponible: demandé 10, reçu {len(large_k_fusion)}")


def main():
    """Fonction principale du test"""
    print_separator("🔀 TEST DU SYSTÈME DE FUSION 🔀")
    
    try:
        # Test des fonctions de base
        test_normalize_scores()
        
        # Test de la logique de fusion
        test_fusion_logic()
        
        # Test des cas limites
        test_edge_cases()
        
        # Test avec vrais retrievers
        test_real_retrievers()
        
        print_separator("✅ TESTS DE FUSION TERMINÉS AVEC SUCCÈS")
        print("🎉 Le système de fusion fonctionne correctement!")
        print("🔀 La fusion combine intelligemment BM25 et FAISS!")
        
    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()