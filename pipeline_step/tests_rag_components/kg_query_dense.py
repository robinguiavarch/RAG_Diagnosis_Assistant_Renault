#!/usr/bin/env python3
"""
Script graph_retrieval_13.py - Interface pour tester le contexte Knowledge Graph Dense

Objectif : Permettre à l'utilisateur de saisir une requête et voir exactement 
le contexte structuré qui sera fourni au LLM dans le pipeline RAG.

Usage:

# Test d'une requête spécifique
./docker-commands.sh test_graph_retrieval --query "ACAL-006 TPE operation error"

# Mode interactif
./docker-commands.sh test_graph_retrieval --interactive

# Mode batch
./docker-commands.sh test_graph_retrieval --batch

# Voir le menu complet
./docker-commands.sh menu

"""

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from core.retrieval_graph.dense_kg_querier import DenseKGQuerier
    kg_dense_query = DenseKGQuerier()
    
except ImportError as e:
    print(f"❌ Erreur d'import kg_dense_query: {e}")
    print("💡 Vérifiez que kg_dense_query.py est dans src/graph_retrieval/")
    sys.exit(1)

def display_banner():
    """Affiche la bannière du script"""
    print("🔍 GRAPH RETRIEVAL TESTER - Knowledge Base Dense")
    print("=" * 60)
    print("🎯 Objectif : Tester le contexte fourni au LLM")
    print("📋 Source : Knowledge Graph Dense via kg_dense_query.py")
    print()

def display_system_info():
    """Affiche les informations système"""
    print("📊 INFORMATIONS SYSTÈME :")
    try:
        # Test de disponibilité des composants
        config = kg_dense_query.load_settings()
        print(f"   ✅ Configuration chargée")
        print(f"   📦 Modèle embedding : {config['models']['embedding_model']}")
        print(f"   🎯 Seuil similarité : {config['retrieval']['symptom_similarity_threshold']}")
        print(f"   🔢 Top-K symptômes : {config['retrieval']['symptom_top_k']}")
        
        # Test Neo4j
        stats = kg_dense_query.get_kb_stats()
        print(f"   ✅ Neo4j Dense accessible")
        print(f"   📈 Symptômes dans KB : {stats['symptoms']}")
        print(f"   📈 Causes dans KB : {stats['causes']}")
        print(f"   📈 Remèdes dans KB : {stats['remedies']}")
        
        # Test FAISS
        try:
            index, metadata = kg_dense_query.load_symptom_index()
            print(f"   ✅ Index FAISS disponible")
            print(f"   📊 Symptômes indexés : {index.ntotal}")
            print(f"   🧠 Dimension embeddings : {index.d}")
        except FileNotFoundError:
            print(f"   ⚠️  Index FAISS non trouvé - Lancez create_symptom_faiss_dense_11.py")
            
    except Exception as e:
        print(f"   ❌ Erreur système : {e}")
    print()

def format_context_for_display(context: str, query: str) -> str:
    """Formate le contexte pour un affichage clair"""
    separator = "=" * 80
    
    formatted = f"""
{separator}
🤖 CONTEXTE QUI SERA FOURNI AU LLM
{separator}

📝 REQUÊTE UTILISATEUR :
"{query}"

📋 CONTEXTE STRUCTURÉ EXTRAIT :
{'-' * 40}
{context}
{'-' * 40}

💡 ANALYSE DU CONTEXTE :
• Ce contexte sera injecté dans le prompt du LLM
• Le LLM utilisera ces informations structurées pour générer sa réponse
• Plus le contexte est riche, plus la réponse sera précise et actionnable

{separator}
"""
    return formatted

def analyze_context_quality(context: str) -> dict:
    """Analyse la qualité du contexte retourné"""
    analysis = {
        'has_content': 'No relevant' not in context and 'Error' not in context,
        'triplet_count': context.count('Triplet ') if 'Triplet ' in context else 0,
        'symptom_count': context.count('Symptom:'),
        'cause_count': context.count('Cause:'),
        'remedy_count': context.count('Remedy:'),
        'avg_similarity': 0.0
    }
    
    # Extraction des scores de similarité
    import re
    similarity_scores = re.findall(r'similarité: ([\d.]+)', context)
    if similarity_scores:
        scores = [float(score) for score in similarity_scores]
        analysis['avg_similarity'] = sum(scores) / len(scores)
    
    return analysis

def display_context_analysis(analysis: dict):
    """Affiche l'analyse du contexte"""
    print("🔬 ANALYSE DE LA QUALITÉ DU CONTEXTE :")
    print(f"   📊 Contexte trouvé : {'✅ Oui' if analysis['has_content'] else '❌ Non'}")
    print(f"   🔢 Triplets trouvés : {analysis['triplet_count']}")
    print(f"   🎯 Symptômes matchés : {analysis['symptom_count']}")
    print(f"   🔍 Causes identifiées : {analysis['cause_count']}")
    print(f"   💡 Remèdes suggérés : {analysis['remedy_count']}")
    if analysis['avg_similarity'] > 0:
        print(f"   📈 Similarité moyenne : {analysis['avg_similarity']:.3f}")
    
    # Recommandations
    print("\n💭 RECOMMANDATIONS :")
    if not analysis['has_content']:
        print("   ⚠️  Aucun contexte trouvé - Essayez une requête plus spécifique")
    elif analysis['triplet_count'] < 2:
        print("   💡 Peu de triplets - La requête pourrait être plus générale")
    elif analysis['triplet_count'] > 5:
        print("   ✅ Bon contexte - Le LLM aura suffisamment d'informations")
    else:
        print("   ✅ Contexte équilibré - Bonne base pour la génération LLM")

def test_single_query(query: str, format_type: str = "detailed", show_analysis: bool = True):
    """Teste une requête unique et affiche le résultat"""
    print(f"🔍 Test de la requête : '{query}'")
    print(f"📋 Format demandé : {format_type}")
    print()
    
    try:
        # Récupération du contexte
        print("⏳ Recherche en cours...")
        context = kg_dense_query.get_structured_context(query, format_type)
        
        # Affichage du contexte formaté
        formatted_context = format_context_for_display(context, query)
        print(formatted_context)
        
        # Analyse de la qualité si demandée
        if show_analysis and format_type == "detailed":
            analysis = analyze_context_quality(context)
            display_context_analysis(analysis)
        
        return context
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du contexte : {e}")
        return None

def interactive_mode():
    """Mode interactif pour tester plusieurs requêtes"""
    print("🎮 MODE INTERACTIF ACTIVÉ")
    print("💡 Tapez 'quit' ou 'exit' pour sortir")
    print("💡 Tapez 'help' pour voir les commandes")
    print()
    
    while True:
        try:
            # Saisie utilisateur
            user_input = input("🔍 Entrez votre requête (ou 'help'): ").strip()
            
            if not user_input:
                continue
                
            # Commandes spéciales
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break
                
            elif user_input.lower() in ['help', 'h']:
                print("\n📋 COMMANDES DISPONIBLES :")
                print("   help          - Afficher cette aide")
                print("   quit/exit     - Quitter le mode interactif")
                print("   stats         - Afficher les statistiques système")
                print("   compact       - Format compact pour la prochaine requête")
                print("   detailed      - Format détaillé pour la prochaine requête")
                print("   json          - Format JSON pour la prochaine requête")
                print("   <votre_requête> - Tester une requête symptôme")
                print()
                continue
                
            elif user_input.lower() == 'stats':
                display_system_info()
                continue
                
            elif user_input.lower() in ['compact', 'detailed', 'json']:
                print(f"📋 Format défini sur : {user_input}")
                continue
            
            # Test de la requête
            print("-" * 60)
            test_single_query(user_input, format_type="detailed", show_analysis=True)
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Interruption utilisateur - Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")

def batch_test_mode(queries: list):
    """Mode batch pour tester plusieurs requêtes prédéfinies"""
    print(f"🚀 MODE BATCH - Test de {len(queries)} requêtes")
    print()
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"📝 Test {i}/{len(queries)}")
        context = test_single_query(query, format_type="compact", show_analysis=False)
        results.append({
            'query': query,
            'context': context,
            'success': context is not None and 'No relevant' not in context
        })
        print("\n" + "="*80 + "\n")
    
    # Résumé des résultats
    print("📊 RÉSUMÉ DES TESTS BATCH :")
    success_count = sum(1 for r in results if r['success'])
    print(f"   ✅ Requêtes avec contexte : {success_count}/{len(queries)}")
    print(f"   ❌ Requêtes sans contexte : {len(queries) - success_count}/{len(queries)}")
    
    # Détail des échecs
    failures = [r for r in results if not r['success']]
    if failures:
        print("\n⚠️  REQUÊTES SANS CONTEXTE :")
        for f in failures:
            print(f"   • '{f['query']}'")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Testeur de contexte Knowledge Graph Dense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/graph_retrieval_13.py
  python scripts/graph_retrieval_13.py --query "ACAL-006 TPE operation error"
  python scripts/graph_retrieval_13.py --interactive
  python scripts/graph_retrieval_13.py --batch
  python scripts/graph_retrieval_13.py --query "robot not working" --format json
        """
    )
    
    parser.add_argument("--query", "-q", type=str, 
                       help="Requête à tester (mode single)")
    parser.add_argument("--format", "-f", choices=["detailed", "compact", "json"],
                       default="detailed", help="Format de sortie du contexte")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Mode interactif")
    parser.add_argument("--batch", "-b", action="store_true",
                       help="Mode batch avec requêtes prédéfinies")
    parser.add_argument("--no-analysis", action="store_true",
                       help="Désactiver l'analyse de qualité")
    parser.add_argument("--no-banner", action="store_true",
                       help="Masquer la bannière")
    
    args = parser.parse_args()
    
    # Affichage de la bannière
    if not args.no_banner:
        display_banner()
        display_system_info()
    
    # Mode de fonctionnement
    if args.query:
        # Mode requête unique
        test_single_query(
            args.query, 
            format_type=args.format, 
            show_analysis=not args.no_analysis
        )
        
    elif args.interactive:
        # Mode interactif
        interactive_mode()
        
    elif args.batch:
        # Mode batch avec requêtes prédéfinies
        test_queries = [
            "ACAL-006 TPE operation error",
            "motor overheating", 
            "engine not starting",
            "robot malfunction",
            "temperature too high",
            "power failure",
            "conveyor belt stuck",
            "sensor error",
            "hydraulic leak",
            "brake not working",
            "communication error"
        ]
        batch_test_mode(test_queries)
        
    else:
        # Mode par défaut : interactif
        interactive_mode()

if __name__ == "__main__":
    main()