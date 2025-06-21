#!/usr/bin/env python3
"""
Test du BM25Retriever lexical
Teste les fonctionnalités de recherche et affiche les résultats détaillés
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from core.retrieval_engine.lexical_search import BM25Retriever


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_separator(title: str, char: str = "="):
    """Affiche un séparateur avec titre"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_result(result: Dict, index: int):
    """Affiche un résultat de recherche de manière formatée"""
    print(f"\n📄 RÉSULTAT #{index + 1}")
    print(f"🆔 Document: {result['document_id']}")
    print(f"🧩 Chunk: {result['chunk_id']}")
    print(f"📊 Score BM25: {result['score']:.4f}")
    print(f"📝 Mots: {result.get('word_count', 'N/A')}")
    print(f"🔤 Caractères: {result.get('char_count', 'N/A')}")
    print(f"⭐ Qualité: {result.get('quality_score', 'N/A')}")
    print(f"📂 Source: {result.get('source_file', 'N/A')}")
    print(f"🔧 Méthode: {result.get('chunking_method', 'N/A')}")
    print(f"📖 TEXTE:")
    print("-" * 50)
    # Afficher le texte avec des retours à la ligne pour la lisibilité
    text = result['text']
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > 80:  # 80 caractères par ligne
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                lines.append(word)  # Mot très long
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines:
        print(line)
    print("-" * 50)


def test_basic_functionality():
    """Test les fonctionnalités de base du retriever"""
    print_separator("TEST DES FONCTIONNALITÉS DE BASE")
    
    try:
        # Chargement de la configuration
        settings = load_settings()
        index_dir = Path(settings["paths"]["bm25_index"])
        
        print(f"📁 Index BM25: {index_dir}")
        
        # Vérification de l'existence de l'index
        if not index_dir.exists():
            print(f"❌ Index BM25 non trouvé: {index_dir}")
            print("💡 Exécutez d'abord: poetry run python scripts/04_index_bm25.py")
            return None
        
        # Initialisation du retriever (plus simple maintenant)
        print("\n🔄 Initialisation du BM25Retriever...")
        retriever = BM25Retriever(index_dir=index_dir)
        print("✅ Retriever initialisé avec succès")
        
        # Statistiques de l'index
        stats = retriever.get_document_stats()
        print(f"\n📊 STATISTIQUES DE L'INDEX:")
        print(f"   📄 Total chunks: {stats['total_chunks']}")
        print(f"   📋 Documents uniques: {stats['unique_documents']}")
        print(f"   📈 Chunks/document (moy): {stats['avg_chunks_per_doc']:.1f}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_search_queries(retriever: BM25Retriever):
    """Test différentes requêtes de recherche"""
    print_separator("TEST DES REQUÊTES DE RECHERCHE")
    
    # Requêtes de test
    test_queries = [
        {
            "query": "I got the error ACAL-006 TPE operation error on the FANUC teach pendant. What should I do?",
            "description": "Requête principale - Erreur FANUC ACAL-006",
            "top_k": 5
        },
        {
            "query": "FANUC error ACAL-006",
            "description": "Requête simplifiée - Mots-clés FANUC",
            "top_k": 3
        },
        {
            "query": "TPE operation error teach pendant",
            "description": "Requête technique - TPE teach pendant",
            "top_k": 3
        },
        {
            "query": "robot calibration error",
            "description": "Requête générale - Erreur calibration",
            "top_k": 3
        },
        {
            "query": "diagnostic troubleshooting",
            "description": "Requête générale - Diagnostic",
            "top_k": 3
        }
    ]
    
    for i, test_case in enumerate(test_queries):
        print_separator(f"REQUÊTE {i+1}: {test_case['description']}", "-")
        print(f"🔍 Requête: \"{test_case['query']}\"")
        print(f"📊 Top-K: {test_case['top_k']}")
        
        try:
            # Recherche normale
            results = retriever.search(
                query=test_case['query'], 
                top_k=test_case['top_k'],
                min_score=0.1  # Score minimum pour filtrer les résultats peu pertinents
            )
            
            if not results:
                print("❌ Aucun résultat trouvé")
                
                # Essayer une recherche debug pour comprendre
                debug_info = retriever.debug_search(test_case['query'], top_k=1)
                print(f"🐛 Debug - requête nettoyée: \"{debug_info['cleaned_query']}\"")
                print(f"🐛 Debug - stats index: {debug_info['index_stats']}")
                continue
            
            print(f"✅ {len(results)} résultat(s) trouvé(s)")
            
            # Affichage des résultats
            for j, result in enumerate(results):
                print_result(result, j)
            
            # Analyse de la pertinence
            scores = [r['score'] for r in results]
            print(f"\n📈 ANALYSE DES SCORES:")
            print(f"   🎯 Score max: {max(scores):.4f}")
            print(f"   📊 Score min: {min(scores):.4f}")
            print(f"   📈 Score moyen: {sum(scores)/len(scores):.4f}")
            
        except Exception as e:
            print(f"❌ Erreur recherche: {e}")
            import traceback
            traceback.print_exc()


def test_edge_cases(retriever: BM25Retriever):
    """Test des cas limites"""
    print_separator("TEST DES CAS LIMITES")
    
    edge_cases = [
        "",  # Requête vide
        "   ",  # Espaces seulement
        "azertyuiopqsdfghjklm",  # Mot inexistant
        "a",  # Requête très courte
        "the and or in on at",  # Mots courants seulement
        "error!!! ???",  # Caractères spéciaux
        "FANUC FANUC FANUC error error error",  # Répétitions
    ]
    
    for i, query in enumerate(edge_cases):
        print(f"\n🧪 Cas limite {i+1}: \"{query}\"")
        try:
            results = retriever.search(query, top_k=2)
            print(f"   📊 Résultats: {len(results)}")
            if results:
                print(f"   🎯 Meilleur score: {results[0]['score']:.4f}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")


def test_performance(retriever: BM25Retriever):
    """Test de performance basique"""
    print_separator("TEST DE PERFORMANCE")
    
    import time
    
    query = "FANUC error ACAL-006 TPE operation"
    num_searches = 10
    
    print(f"🏃 Test de {num_searches} recherches avec: \"{query}\"")
    
    start_time = time.time()
    
    for i in range(num_searches):
        results = retriever.search(query, top_k=5)
        if i == 0:
            first_result_count = len(results)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_searches
    
    print(f"⏱️ Temps total: {total_time:.3f}s")
    print(f"⚡ Temps moyen par recherche: {avg_time:.3f}s")
    print(f"📊 {first_result_count} résultats par recherche")
    print(f"🚀 Recherches/seconde: {num_searches/total_time:.1f}")


def main():
    """Fonction principale du test"""
    print_separator("🧪 TEST DU BM25RETRIEVER LEXICAL 🧪")
    
    try:
        # Test des fonctionnalités de base
        retriever = test_basic_functionality()
        
        if retriever is None:
            print("❌ Impossible d'initialiser le retriever. Arrêt des tests.")
            return
        
        # Test des requêtes de recherche
        test_search_queries(retriever)
        
        # Test des cas limites
        test_edge_cases(retriever)
        
        # Test de performance
        test_performance(retriever)
        
        print_separator("✅ TESTS TERMINÉS AVEC SUCCÈS")
        print("🎉 Le BM25Retriever fonctionne correctement!")
        
    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()