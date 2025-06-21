"""
Script principal de retrieval hybride BM25 + FAISS avec option re-ranking
Combine recherche lexicale, sémantique et re-ranking pour des résultats optimaux

IMPORTANT: Pour utiliser ce script:

# Mode interactif (recommandé pour débuter)
poetry run python scripts/06_retrieval.py -i

# Requête rapide
poetry run python scripts/06_retrieval.py -q "robot calibration error"

# Avec re-ranking CrossEncoder
poetry run python scripts/06_retrieval.py -q "FANUC error" --rerank

# Avec détails des scores
poetry run python scripts/06_retrieval.py -q "FANUC error" --details --rerank
"""

import os
# Variables d'environnement pour éviter les conflits sur macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import argparse
from datetime import datetime
import sys
import time

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent.parent)) 

from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever
from core.retrieval_engine.hybrid_fusion import fuse_results

# Import optionnel du reranker
try:
    from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
    RERANKER_AVAILABLE = True
except ImportError:
    print("⚠️ CrossEncoderReranker non disponible")
    RERANKER_AVAILABLE = False
    CrossEncoderReranker = None


def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def initialize_retrievers(settings: Dict[str, Any]) -> tuple[BM25Retriever, FAISSRetriever]:
    """Initialise les retrievers BM25 et FAISS"""
    print("🔍 Initialisation des retrievers...")
    
    # BM25 Retriever
    bm25_index_dir = Path(settings["paths"]["bm25_index"])
    if not bm25_index_dir.exists():
        raise FileNotFoundError(
            f"Index BM25 non trouvé: {bm25_index_dir}. "
            f"Exécutez: poetry run python scripts/04_index_bm25.py"
        )
    
    bm25 = BM25Retriever(index_dir=bm25_index_dir)
    print("✅ BM25Retriever initialisé")
    
    # FAISS Retriever
    faiss_index_dir = Path(settings["paths"]["faiss_index"])
    index_path = faiss_index_dir / "index.faiss"
    metadata_path = faiss_index_dir / "metadata.pkl"
    
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index FAISS non trouvé: {faiss_index_dir}. "
            f"Exécutez: poetry run python scripts/05_create_faiss_index.py"
        )
    
    faiss = FAISSRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model_name=settings["models"]["embedding_model"]
    )
    print("✅ FAISSRetriever initialisé")
    
    return bm25, faiss


def initialize_reranker(settings: Dict[str, Any]) -> Optional[CrossEncoderReranker]:
    """Initialise le reranker si disponible et activé"""
    if not RERANKER_AVAILABLE:
        print("⚠️ Reranker non disponible (module manquant)")
        return None
    
    reranking_config = settings.get("reranking", {})
    if not reranking_config.get("enabled", False):
        print("ℹ️ Reranker désactivé dans la configuration")
        return None
    
    try:
        reranker_model = settings["models"]["reranker_model"]
        reranker = CrossEncoderReranker(model_name=reranker_model)
        print("✅ CrossEncoderReranker initialisé")
        return reranker
    except Exception as e:
        print(f"❌ Erreur initialisation reranker: {e}")
        return None


def search_and_fuse(
    bm25: BM25Retriever, 
    faiss: FAISSRetriever, 
    query: str, 
    top_k: int = 10,
    reranker: Optional[CrossEncoderReranker] = None,
    settings: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Effectue la recherche hybride, fusionne et optionnellement re-rank les résultats"""
    
    if not query.strip():
        print("❌ Requête vide")
        return []
    
    print(f"🔍 Requête: \"{query}\"")
    
    # Configuration du retrieval
    if settings:
        retrieval_config = settings.get("retrieval", {})
        reranking_config = settings.get("reranking", {})
        top_k_sparse = retrieval_config.get("top_k_sparse", top_k * 2)
        top_k_dense = retrieval_config.get("top_k_dense", top_k * 2)
        top_k_before_rerank = reranking_config.get("top_k_before_rerank", top_k * 2)
    else:
        top_k_sparse = top_k * 2
        top_k_dense = top_k * 2
        top_k_before_rerank = top_k * 2
    
    print(f"📊 Config: BM25={top_k_sparse}, FAISS={top_k_dense}, Final={top_k}")
    
    # Recherche BM25
    print("🔎 Recherche lexicale (BM25)...")
    start_time = time.time()
    try:
        bm25_results = bm25.search(query, top_k=top_k_sparse)
        bm25_time = time.time() - start_time
        print(f"   📝 BM25: {len(bm25_results)} résultats ({bm25_time:.2f}s)")
    except Exception as e:
        print(f"❌ Erreur BM25: {e}")
        bm25_results = []
        bm25_time = 0
    
    # Recherche FAISS
    print("🧠 Recherche sémantique (FAISS)...")
    start_time = time.time()
    try:
        faiss_results = faiss.search(query, top_k=top_k_dense)
        faiss_time = time.time() - start_time
        print(f"   🧠 FAISS: {len(faiss_results)} résultats ({faiss_time:.2f}s)")
    except Exception as e:
        print(f"❌ Erreur FAISS: {e}")
        faiss_results = []
        faiss_time = 0
    
    # Vérifier qu'on a des résultats
    if not bm25_results and not faiss_results:
        print("❌ Aucun résultat trouvé dans les deux retrievers")
        return []
    
    # Fusion
    print("⚖️ Fusion des résultats...")
    start_time = time.time()
    fused_results = fuse_results(bm25_results, faiss_results, top_k=top_k_before_rerank)
    fusion_time = time.time() - start_time
    print(f"   🔀 Fusion: {len(fused_results)} résultats ({fusion_time:.2f}s)")
    
    # Re-ranking optionnel
    if reranker and fused_results:
        print("🎯 Re-ranking avec CrossEncoder...")
        start_time = time.time()
        try:
            final_results = reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=top_k
            )
            rerank_time = time.time() - start_time
            print(f"   🏆 Re-ranking: {len(final_results)} résultats finaux ({rerank_time:.2f}s)")
            
            # Ajouter les temps de traitement
            for result in final_results:
                result["processing_times"] = {
                    "bm25_time": bm25_time,
                    "faiss_time": faiss_time,
                    "fusion_time": fusion_time,
                    "rerank_time": rerank_time,
                    "total_time": bm25_time + faiss_time + fusion_time + rerank_time
                }
            
            return final_results
            
        except Exception as e:
            print(f"❌ Erreur re-ranking: {e}")
            print("🔄 Fallback: résultats de fusion")
            return fused_results[:top_k]
    else:
        if reranker is None and settings and settings.get("reranking", {}).get("enabled", False):
            print("⚠️ Re-ranking activé mais reranker non disponible")
        
        return fused_results[:top_k]


def display_results(results: List[Dict[str, Any]], show_details: bool = False, show_rerank: bool = False):
    """Affiche les résultats de manière formatée avec support re-ranking"""
    if not results:
        print("❌ Aucun résultat à afficher")
        return
    
    # Détection si re-ranking utilisé
    has_rerank_scores = any("cross_encoder_score" in doc for doc in results)
    
    title = "📄 TOP-{} DOCUMENTS PERTINENTS{}:".format(
        len(results), 
        " (RE-RANKÉS)" if has_rerank_scores else ""
    )
    print(f"\n{title}")
    print("=" * 70)
    
    for i, doc in enumerate(results, 1):
        print(f"\n🏆 RÉSULTAT #{i}")
        
        # Affichage des scores selon disponibilité
        if has_rerank_scores and "cross_encoder_score" in doc:
            print(f"🏅 Score CrossEncoder: {doc['cross_encoder_score']:.4f}")
        
        print(f"🎯 Score fusionné: {doc.get('fused_score', 0):.4f}")
        
        if show_details:
            print(f"📝 Score BM25: {doc.get('bm25_score', 0):.4f}")
            print(f"🧠 Score FAISS: {doc.get('faiss_score', 0):.4f}")
            
            if "original_rank" in doc:
                print(f"📊 Rang original: #{doc['original_rank']}")
            
            if "processing_times" in doc:
                times = doc["processing_times"]
                print(f"⏱️ Temps total: {times['total_time']:.3f}s")
        
        print(f"🆔 Document: {doc['document_id']}")
        print(f"🧩 Chunk: {doc['chunk_id']}")
        print(f"📖 TEXTE:")
        print("-" * 50)
        
        # Affichage du texte avec retours à la ligne
        text = doc['text']
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 80:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            print(line)
        
        print("-" * 50)


def save_results(
    results: List[Dict[str, Any]], 
    query: str, 
    output_dir: Path, 
    filename: str = None,
    has_reranking: bool = False
) -> Path:
    """Sauvegarde les résultats avec métadonnées"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "reranked_results" if has_reranking else "retrieval_results"
        filename = f"{prefix}_{timestamp}.json"
    
    output_path = output_dir / filename
    
    # Structure complète avec métadonnées
    method = "hybrid_bm25_faiss_crossencoder" if has_reranking else "hybrid_bm25_faiss"
    stages = ["bm25", "faiss", "fusion"]
    if has_reranking:
        stages.append("cross_encoder_rerank")
    
    metadata = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "retrieval_method": method,
        "pipeline_stages": stages,
        "has_reranking": has_reranking
    }
    
    # Ajouter stats de performance si disponibles
    if results and "processing_times" in results[0]:
        times = results[0]["processing_times"]
        metadata["performance"] = {
            "total_time_seconds": times["total_time"],
            "bm25_time_seconds": times["bm25_time"],
            "faiss_time_seconds": times["faiss_time"],
            "fusion_time_seconds": times["fusion_time"]
        }
        if "rerank_time" in times:
            metadata["performance"]["rerank_time_seconds"] = times["rerank_time"]
    
    output_data = {
        "metadata": metadata,
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path


def interactive_mode(
    bm25: BM25Retriever, 
    faiss: FAISSRetriever, 
    settings: Dict[str, Any],
    reranker: Optional[CrossEncoderReranker] = None
):
    """Mode interactif pour tester plusieurs requêtes"""
    print("\n🎯 MODE INTERACTIF")
    print("=" * 50)
    print("Entrez vos requêtes (tapez 'quit' pour quitter)")
    
    if reranker:
        print("🏅 Re-ranking CrossEncoder activé")
    
    print("💡 Commandes spéciales:")
    print("  'details' - Activer/désactiver les détails")
    print("  'rerank on/off' - Activer/désactiver le re-ranking")
    
    output_dir = Path(settings["paths"].get("outputs", "outputs/retrieval"))
    show_details = False
    use_reranker = reranker is not None
    
    while True:
        try:
            user_input = input("\n❓ Requête: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if user_input.lower() == 'details':
                show_details = not show_details
                print(f"🔧 Mode détails: {'ON' if show_details else 'OFF'}")
                continue
            
            if user_input.lower() in ['rerank on', 'rerank off']:
                if user_input.lower() == 'rerank on' and reranker:
                    use_reranker = True
                    print("🏅 Re-ranking activé")
                else:
                    use_reranker = False
                    print("📊 Re-ranking désactivé")
                continue
            
            if not user_input:
                print("⚠️ Requête vide, essayez autre chose")
                continue
            
            # Recherche et fusion avec re-ranking optionnel
            current_reranker = reranker if use_reranker else None
            results = search_and_fuse(bm25, faiss, user_input, top_k=5, reranker=current_reranker, settings=settings)
            
            if results:
                # Affichage
                has_rerank_scores = any("cross_encoder_score" in doc for doc in results)
                display_results(results, show_details=show_details, show_rerank=has_rerank_scores)
                
                # Sauvegarde optionnelle
                save_choice = input("\n💾 Sauvegarder ces résultats? (y/N): ").strip().lower()
                if save_choice in ['y', 'yes', 'oui']:
                    output_path = save_results(results, user_input, output_dir, has_reranking=has_rerank_scores)
                    print(f"✅ Résultats sauvegardés: {output_path}")
            else:
                print("❌ Aucun résultat trouvé")
                
        except KeyboardInterrupt:
            print("\n👋 Interruption utilisateur, au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")


def batch_mode(
    bm25: BM25Retriever, 
    faiss: FAISSRetriever, 
    queries: List[str], 
    settings: Dict[str, Any],
    reranker: Optional[CrossEncoderReranker] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Mode batch pour traiter plusieurs requêtes"""
    print(f"\n📋 MODE BATCH - {len(queries)} requêtes")
    if reranker:
        print("🏅 Avec re-ranking CrossEncoder")
    print("=" * 50)
    
    output_dir = Path(settings["paths"].get("outputs", "outputs/retrieval"))
    all_results = {}
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔄 Requête {i}/{len(queries)}")
        results = search_and_fuse(bm25, faiss, query, top_k=10, reranker=reranker, settings=settings)
        all_results[query] = results
        
        if results:
            print(f"✅ {len(results)} résultats trouvés")
        else:
            print("❌ Aucun résultat")
    
    # Sauvegarde globale
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "batch_reranked" if reranker else "batch_results"
    batch_output = output_dir / f"{prefix}_{timestamp}.json"
    
    method = "hybrid_bm25_faiss_crossencoder" if reranker else "hybrid_bm25_faiss"
    
    batch_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries),
            "retrieval_method": method,
            "has_reranking": reranker is not None
        },
        "results": all_results
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(batch_output, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Résultats batch sauvegardés: {batch_output}")
    return all_results


def main():
    """Fonction principale avec support arguments ligne de commande"""
    parser = argparse.ArgumentParser(description="Retrieval hybride BM25 + FAISS avec option re-ranking")
    parser.add_argument("--config", default="config/settings.yaml", help="Chemin config YAML")
    parser.add_argument("--query", "-q", help="Requête unique")
    parser.add_argument("--interactive", "-i", action="store_true", help="Mode interactif")
    parser.add_argument("--batch", help="Fichier texte avec requêtes (une par ligne)")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de résultats (défaut: 5)")
    parser.add_argument("--details", action="store_true", help="Afficher scores détaillés")
    parser.add_argument("--output", help="Répertoire de sortie personnalisé")
    parser.add_argument("--rerank", action="store_true", help="Activer le re-ranking CrossEncoder")
    parser.add_argument("--no-rerank", action="store_true", help="Forcer désactivation du re-ranking")
    
    args = parser.parse_args()
    
    try:
        print("🚀 RETRIEVAL HYBRIDE BM25 + FAISS")
        if args.rerank and RERANKER_AVAILABLE:
            print("🏅 Avec option re-ranking CrossEncoder")
        print("=" * 60)
        
        # Chargement configuration
        print("🔧 Chargement des paramètres...")
        settings = load_settings(args.config)
        
        # Override output si spécifié
        if args.output:
            settings["paths"]["outputs"] = args.output
        
        # Gestion du re-ranking
        if args.no_rerank:
            settings["reranking"]["enabled"] = False
        elif args.rerank:
            settings["reranking"]["enabled"] = True
        
        # Initialisation retrievers
        bm25, faiss = initialize_retrievers(settings)
        
        # Initialisation reranker optionnel
        reranker = None
        if (args.rerank or settings.get("reranking", {}).get("enabled", False)) and not args.no_rerank:
            reranker = initialize_reranker(settings)
        
        # Statistiques
        bm25_stats = bm25.get_document_stats()
        faiss_stats = faiss.get_index_stats()
        
        print(f"\n📊 STATISTIQUES:")
        print(f"   📝 BM25: {bm25_stats['total_chunks']} chunks, {bm25_stats['unique_documents']} docs")
        print(f"   🧠 FAISS: {faiss_stats['total_vectors']} vecteurs, dim={faiss_stats['vector_dimension']}")
        
        if reranker:
            reranker_info = reranker.get_model_info()
            print(f"   🏅 Reranker: {reranker_info['model_name']} sur {reranker_info['device']}")
        
        # Mode selon arguments
        if args.batch:
            # Mode batch
            print(f"📋 Chargement requêtes depuis: {args.batch}")
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                print(f"❌ Fichier de requêtes non trouvé: {args.batch}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Erreur lecture fichier de requêtes: {e}")
                sys.exit(1)
                
            batch_mode(bm25, faiss, queries, settings, reranker=reranker)
            
        elif args.query:
            # Requête unique
            results = search_and_fuse(bm25, faiss, args.query, top_k=args.top_k, reranker=reranker, settings=settings)
            
            if results:
                has_rerank_scores = any("cross_encoder_score" in doc for doc in results)
                display_results(results, show_details=args.details, show_rerank=has_rerank_scores)
                
                output_dir = Path(settings["paths"].get("outputs", "outputs/retrieval"))
                output_path = save_results(results, args.query, output_dir, has_reranking=has_rerank_scores)
                print(f"\n✅ Résultats sauvegardés: {output_path}")
            else:
                print("❌ Aucun résultat trouvé")
            
        else:
            # Mode interactif par défaut
            interactive_mode(bm25, faiss, settings, reranker=reranker)
    
    except FileNotFoundError as e:
        print(f"❌ Fichier manquant: {e}")
        print("💡 Assurez-vous d'avoir exécuté les scripts de préparation des index")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interruption utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()