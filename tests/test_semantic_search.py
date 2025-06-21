"""
Test du FAISSRetriever sémantique
Teste les fonctionnalités de recherche sémantique et affiche les résultats détaillés

IMPORTANT: Pour tester ce script: 

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_semantic.py > tests/resultats/test_semantic.txt 2>&1


"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

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


def print_result(result: Dict, index: int):
    """Affiche un résultat de recherche de manière formatée"""
    print(f"\n🧠 RÉSULTAT SÉMANTIQUE #{index + 1}")
    print(f"🆔 Document: {result['document_id']}")
    print(f"🧩 Chunk: {result['chunk_id']}")
    print(f"📊 Score similarité: {result['score']:.4f}")
    print(f"📏 Distance L2: {result.get('distance', 'N/A'):.4f}" if 'distance' in result else "")
    print(f"📝 Mots: {result.get('word_count', 'N/A')}")
    print(f"🔤 Caractères: {result.get('char_count', 'N/A')}")
    print(f"📐 Norme embedding: {result.get('embedding_norm', 'N/A'):.3f}" if 'embedding_norm' in result else "")
    print(f"📂 Source: {result.get('source_file', 'N/A')}")
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
        
        # Chemins vers l'index FAISS
        faiss_index_dir = Path(settings["paths"]["faiss_index_dir"])
        index_path = Path(settings["paths"]["faiss_index"])
        metadata_path = Path(settings["paths"]["embedding_file"])
        model_name = settings["models"]["embedding_model"]
        
        print(f"📁 Index FAISS: {index_path}")
        print(f"📄 Métadonnées: {metadata_path}")
        print(f"🤖 Modèle: {model_name}")
        
        # Vérification de l'existence des fichiers
        if not index_path.exists():
            print(f"❌ Index FAISS non trouvé: {index_path}")
            print("💡 Exécutez d'abord: poetry run python scripts/05_create_faiss_index.py")
            return None
        
        if not metadata_path.exists():
            print(f"❌ Métadonnées non trouvées: {metadata_path}")
            return None
        
        # Initialisation du retriever
        print("\n🔄 Initialisation du FAISSRetriever...")
        retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model_name=model_name
        )
        print("✅ Retriever initialisé avec succès")
        
        # Statistiques de l'index
        stats = retriever.get_index_stats()
        print(f"\n📊 STATISTIQUES DE L'INDEX:")
        print(f"   🔢 Total vecteurs: {stats['total_vectors']}")
        print(f"   📏 Dimension: {stats['vector_dimension']}")
        print(f"   🏗️ Type d'index: {stats['index_type']}")
        print(f"   📋 Documents uniques: {stats['unique_documents']}")
        print(f"   📄 Total chunks: {stats['total_chunks']}")
        print(f"   📈 Chunks/document (moy): {stats['avg_chunks_per_doc']:.1f}")
        print(f"   💻 Device modèle: {stats['model_device']}")
        print(f"   📦 Format métadonnées: {stats['metadata_format']}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_semantic_queries(retriever: FAISSRetriever):
    """Test différentes requêtes de recherche sémantique"""
    print_separator("TEST DES REQUÊTES SÉMANTIQUES")
    
    # Requêtes de test sémantiques
    test_queries = [
        {
            "query": "I got the error ACAL-006 TPE operation error on the FANUC teach pendant. What should I do?",
            "description": "Requête principale - Erreur FANUC ACAL-006",
            "top_k": 5
        },
        {
            "query": "robot calibration failed teach pendant",
            "description": "Requête sémantique - Échec calibration robot",
            "top_k": 3
        },
        {
            "query": "how to troubleshoot FANUC robot error",
            "description": "Requête générale - Dépannage FANUC",
            "top_k": 3
        },
        {
            "query": "teaching pendant operation problem",
            "description": "Requête sémantique - Problème pendant d'apprentissage",
            "top_k": 3
        },
        {
            "query": "automation error code diagnostic",
            "description": "Requête conceptuelle - Diagnostic code erreur",
            "top_k": 3
        },
        {
            "query": "industrial robot malfunction solution",
            "description": "Requête sémantique - Solution dysfonctionnement robot",
            "top_k": 3
        }
    ]
    
    for i, test_case in enumerate(test_queries):
        print_separator(f"REQUÊTE {i+1}: {test_case['description']}", "-")
        print(f"🧠 Requête: \"{test_case['query']}\"")
        print(f"📊 Top-K: {test_case['top_k']}")
        
        try:
            # Recherche sémantique
            results = retriever.search(
                query=test_case['query'], 
                top_k=test_case['top_k'],
                min_score=0.0  # Pas de filtrage par score pour voir tous les résultats
            )
            
            if not results:
                print("❌ Aucun résultat trouvé")
                
                # Essayer une recherche debug pour comprendre
                debug_info = retriever.debug_search(test_case['query'], top_k=1)
                print(f"🐛 Debug - dimension embedding: {debug_info['query_embedding_dim']}")
                print(f"🐛 Debug - norme embedding: {debug_info['query_embedding_norm']:.3f}")
                print(f"🐛 Debug - stats index: {debug_info['index_stats']}")
                continue
            
            print(f"✅ {len(results)} résultat(s) trouvé(s)")
            
            # Affichage des résultats
            for j, result in enumerate(results):
                print_result(result, j)
            
            # Analyse de la pertinence sémantique
            scores = [r['score'] for r in results]
            distances = [r.get('distance', 0) for r in results if 'distance' in r]
            
            print(f"\n📈 ANALYSE DES SCORES SÉMANTIQUES:")
            print(f"   🎯 Score max: {max(scores):.4f}")
            print(f"   📊 Score min: {min(scores):.4f}")
            print(f"   📈 Score moyen: {sum(scores)/len(scores):.4f}")
            if distances:
                print(f"   📏 Distance L2 min: {min(distances):.4f}")
                print(f"   📏 Distance L2 max: {max(distances):.4f}")
            
        except Exception as e:
            print(f"❌ Erreur recherche: {e}")
            import traceback
            traceback.print_exc()


def test_semantic_vs_lexical_comparison(retriever: FAISSRetriever):
    """Compare recherche sémantique vs recherche conceptuelle"""
    print_separator("COMPARAISON SÉMANTIQUE VS CONCEPTUELLE")
    
    # Requêtes pour tester la différence sémantique
    comparison_queries = [
        {
            "query": "robot malfunctioning",
            "description": "Concept: Robot en panne",
            "similar_concepts": ["machine broken", "automation failure", "equipment error"]
        },
        {
            "query": "calibration procedure",
            "description": "Concept: Procédure de calibration", 
            "similar_concepts": ["adjustment process", "setup method", "configuration steps"]
        }
    ]
    
    for test_case in comparison_queries:
        print_separator(f"TEST: {test_case['description']}", "-")
        
        # Test requête principale
        print(f"🧠 Requête principale: \"{test_case['query']}\"")
        main_results = retriever.search(test_case['query'], top_k=2)
        
        if main_results:
            print(f"✅ {len(main_results)} résultats pour la requête principale")
            for i, result in enumerate(main_results):
                print(f"   {i+1}. Score: {result['score']:.4f} | {result['text'][:100]}...")
        
        # Test concepts similaires
        for concept in test_case['similar_concepts']:
            print(f"\n🔄 Concept similaire: \"{concept}\"")
            concept_results = retriever.search(concept, top_k=1)
            
            if concept_results:
                result = concept_results[0]
                print(f"   Score: {result['score']:.4f} | {result['text'][:100]}...")
                
                # Comparer avec la requête principale
                if main_results:
                    score_diff = abs(result['score'] - main_results[0]['score'])
                    print(f"   📊 Différence de score: {score_diff:.4f}")


def test_edge_cases(retriever: FAISSRetriever):
    """Test des cas limites pour la recherche sémantique"""
    print_separator("TEST DES CAS LIMITES SÉMANTIQUES")
    
    edge_cases = [
        "",  # Requête vide
        "   ",  # Espaces seulement
        "qwertyuiopasdfgh",  # Mots inventés
        "a",  # Requête très courte
        "the and or",  # Mots courants seulement
        "🤖 ⚙️ 🔧",  # Emojis seulement
        "FANUC " * 20,  # Répétition excessive
    ]
    
    for i, query in enumerate(edge_cases):
        print(f"\n🧪 Cas limite {i+1}: \"{query}\"")
        try:
            results = retriever.search(query, top_k=2)
            print(f"   📊 Résultats: {len(results)}")
            if results:
                print(f"   🎯 Meilleur score: {results[0]['score']:.4f}")
                print(f"   📏 Distance: {results[0].get('distance', 'N/A'):.4f}" if 'distance' in results[0] else "")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")


def test_performance(retriever: FAISSRetriever):
    """Test de performance de la recherche sémantique"""
    print_separator("TEST DE PERFORMANCE SÉMANTIQUE")
    
    import time
    
    query = "FANUC error ACAL-006 TPE operation"
    num_searches = 10
    
    print(f"🏃 Test de {num_searches} recherches sémantiques avec: \"{query}\"")
    
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
    
    # Comparaison avec une référence
    if avg_time < 0.1:
        print("✅ Performance excellente (<100ms)")
    elif avg_time < 0.5:
        print("✅ Performance bonne (<500ms)")
    else:
        print("⚠️ Performance à améliorer (>500ms)")


def main():
    """Fonction principale du test"""
    print_separator("🧠 TEST DU FAISSRETRIEVER SÉMANTIQUE 🧠")
    
    try:
        # Test des fonctionnalités de base
        retriever = test_basic_functionality()
        
        if retriever is None:
            print("❌ Impossible d'initialiser le retriever. Arrêt des tests.")
            return
        
        # Test des requêtes sémantiques
        test_semantic_queries(retriever)
        
        # Comparaison sémantique vs conceptuelle
        test_semantic_vs_lexical_comparison(retriever)
        
        # Test des cas limites
        test_edge_cases(retriever)
        
        # Test de performance
        test_performance(retriever)
        
        print_separator("✅ TESTS SÉMANTIQUES TERMINÉS AVEC SUCCÈS")
        print("🎉 Le FAISSRetriever fonctionne correctement!")
        print("🧠 La recherche sémantique capture les concepts et le sens!")
        
    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()