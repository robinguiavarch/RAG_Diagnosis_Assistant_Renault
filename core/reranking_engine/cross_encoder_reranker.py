from typing import List, Dict, Union, Any, Optional
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import time  # Ajout de l'import manquant


class CrossEncoderReranker:
    """
    Reranker utilisant un modèle CrossEncoder pour affiner les résultats
    de recherche hybride en évaluant la pertinence query-document.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialise le reranker CrossEncoder
        
        Args:
            model_name: Nom du modèle CrossEncoder à utiliser
            device: Device à utiliser ('cuda', 'cpu', ou None pour auto-détection)
            max_length: Longueur maximale des séquences
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Détermination du device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Chargement CrossEncoder: {model_name}")
        print(f"🎯 Device cible: {device}")
        
        try:
            self.model = CrossEncoder(
                model_name, 
                device=device,
                max_length=max_length
            )
            self.device = device
            print(f"✅ CrossEncoder chargé avec succès sur {device}")
            
        except Exception as e:
            print(f"⚠️ Échec chargement sur {device}: {e}")
            if device == "cuda":
                print("🔄 Tentative fallback sur CPU...")
                try:
                    self.model = CrossEncoder(
                        model_name, 
                        device="cpu",
                        max_length=max_length
                    )
                    self.device = "cpu"
                    print("✅ CrossEncoder chargé sur CPU (fallback)")
                except Exception as e2:
                    raise RuntimeError(f"Impossible de charger le modèle: {e2}")
            else:
                raise RuntimeError(f"Erreur chargement modèle: {e}")

    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5,
        batch_size: int = 32,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Re-rank les candidats selon leur pertinence pour la requête
        
        Args:
            query: Requête utilisateur
            candidates: Liste des candidats à re-ranker
            top_k: Nombre de résultats à retourner
            batch_size: Taille des batches pour le traitement
            return_scores: Si True, inclut tous les scores dans le résultat
            
        Returns:
            Liste des candidats re-rankés avec scores CrossEncoder
        """
        if not candidates:
            print("⚠️ Aucun candidat à re-ranker")
            return []
        
        if not query.strip():
            print("⚠️ Requête vide pour re-ranking, retour résultats originaux")
            return candidates[:top_k]
        
        print(f"🎯 Re-ranking de {len(candidates)} candidats avec CrossEncoder")
        
        # Validation et nettoyage des candidats
        valid_candidates = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("text", "").strip()
            if text:
                # Ajouter l'index original pour traçabilité
                candidate_copy = candidate.copy()
                candidate_copy["original_rank"] = i + 1
                valid_candidates.append(candidate_copy)
            else:
                print(f"⚠️ Candidat {i+1} ignoré (texte vide)")
        
        if not valid_candidates:
            print("❌ Aucun candidat valide pour re-ranking")
            return []
        
        try:
            # Préparation des paires (query, document)
            pairs = []
            for candidate in valid_candidates:
                text = candidate["text"]
                # Troncature si nécessaire pour respecter max_length
                if len(query) + len(text) > self.max_length - 10:  # Marge pour tokens spéciaux
                    # Garder le début du texte qui est souvent le plus important
                    max_text_length = self.max_length - len(query) - 10
                    text = text[:max_text_length]
                
                pairs.append((query, text))
            
            print(f"📊 Scoring {len(pairs)} paires query-document...")
            
            # Calcul des scores CrossEncoder par batches
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores)
            
            # Enrichissement avec les scores CrossEncoder
            for candidate, cross_score in zip(valid_candidates, all_scores):
                candidate["cross_encoder_score"] = float(cross_score)
                
                # Optionnel: garder tous les scores pour analyse
                if return_scores and "fused_score" in candidate:
                    candidate["all_scores"] = {
                        "cross_encoder": float(cross_score),
                        "fusion": candidate["fused_score"],
                        "bm25": candidate.get("bm25_score", 0.0),
                        "faiss": candidate.get("faiss_score", 0.0)
                    }
            
            # Tri par score CrossEncoder (décroissant)
            reranked = sorted(
                valid_candidates,
                key=lambda x: x["cross_encoder_score"],
                reverse=True
            )
            
            print(f"✅ Re-ranking terminé")
            print(f"🏆 Top-1 score: {reranked[0]['cross_encoder_score']:.4f}")
            if len(reranked) > 1:
                print(f"🥈 Top-2 score: {reranked[1]['cross_encoder_score']:.4f}")
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"❌ Erreur durant le re-ranking: {e}")
            print("🔄 Fallback: retour des résultats de fusion originaux")
            
            # Fallback: retourner les candidats triés par score de fusion
            try:
                fallback_results = sorted(
                    valid_candidates,
                    key=lambda x: x.get("fused_score", x.get("score", 0)),
                    reverse=True
                )
                return fallback_results[:top_k]
            except Exception as e2:
                print(f"❌ Erreur fallback: {e2}")
                return valid_candidates[:top_k]

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        """
        Score une liste de paires (query, document)
        
        Args:
            pairs: Liste de tuples (query, document)
            
        Returns:
            Liste des scores de pertinence
        """
        if not pairs:
            return []
        
        try:
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
        except Exception as e:
            print(f"❌ Erreur scoring: {e}")
            return [0.0] * len(pairs)

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle chargé"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "model_type": "CrossEncoder",
            "framework": "sentence-transformers"
        }

    def benchmark_speed(self, query: str, documents: List[str], num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark la vitesse de re-ranking
        
        Args:
            query: Requête de test
            documents: Liste de documents de test
            num_runs: Nombre d'exécutions pour la moyenne
            
        Returns:
            Statistiques de performance
        """
        if not documents:
            return {"error": "Aucun document pour le benchmark"}
        
        pairs = [(query, doc) for doc in documents]
        times = []
        
        print(f"🚀 Benchmark CrossEncoder: {len(documents)} documents, {num_runs} runs")
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                _ = self.model.predict(pairs)
                execution_time = time.time() - start_time
                times.append(execution_time)
                print(f"   Run {run+1}: {execution_time:.3f}s")
            except Exception as e:
                print(f"   Run {run+1}: ERREUR - {e}")
                continue
        
        if not times:
            return {"error": "Tous les runs ont échoué"}
        
        return {
            "avg_time_seconds": np.mean(times),
            "min_time_seconds": np.min(times),
            "max_time_seconds": np.max(times),
            "std_time_seconds": np.std(times),
            "documents_per_second": len(documents) / np.mean(times),
            "device": self.device,
            "num_documents": len(documents),
            "num_runs": len(times)
        }