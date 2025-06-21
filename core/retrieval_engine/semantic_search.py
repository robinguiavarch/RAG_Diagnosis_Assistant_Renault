import pickle
from pathlib import Path
from typing import List, Dict, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialise le retriever FAISS
        
        Args:
            index_path: Chemin vers l'index FAISS (.faiss)
            metadata_path: Chemin vers les métadonnées (.pkl)
            embedding_model_name: Nom du modèle d'embeddings
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Vérifications d'existence
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS non trouvé: {self.index_path}. "
                f"Exécutez d'abord: poetry run python scripts/05_create_faiss_index.py"
            )
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Métadonnées non trouvées: {self.metadata_path}."
            )

        # Charger l'index FAISS
        self.index = faiss.read_index(str(self.index_path))

        # Charger les métadonnées avec support ancien/nouveau format
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
        
        # Détection du format des métadonnées
        if isinstance(data, dict) and "documents" in data:
            # Nouveau format avec métadonnées
            self.documents = data["documents"]
            self.ids = data.get("ids", [])
            self.metadata_info = data.get("metadata", {})
        elif isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
            # Format intermédiaire
            self.documents = data["documents"]
            self.ids = [f"{doc['document_id']}|{doc['chunk_id']}" for doc in self.documents]
            self.metadata_info = {}
        elif isinstance(data, list):
            # Ancien format - liste directe
            self.documents = data
            self.ids = [f"{doc['document_id']}|{doc['chunk_id']}" for doc in data]
            self.metadata_info = {}
        else:
            raise ValueError("Format de métadonnées non reconnu")

        # Validation
        if len(self.documents) != self.index.ntotal:
            raise ValueError(
                f"Incohérence: {len(self.documents)} documents vs {self.index.ntotal} vecteurs dans l'index"
            )

        # Charger le modèle SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(embedding_model_name, device=device)
        
        # Vérification de compatibilité des dimensions
        expected_dim = self.model.get_sentence_embedding_dimension()
        if self.index.d != expected_dim:
            raise ValueError(
                f"Dimension incompatible: modèle={expected_dim}, index={self.index.d}"
            )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Recherche sémantique dans l'index FAISS
        
        Args:
            query: Requête de recherche
            top_k: Nombre maximum de résultats
            min_score: Score minimum (similarité cosinus min)
            
        Returns:
            Liste des chunks trouvés avec métadonnées
        """
        if not query.strip():
            return []
        
        try:
            # Générer l'embedding de la requête (normalisé pour similarité cosinus)
            query_vector = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            query_vector = query_vector.astype("float32")

            # Recherche dans l'index FAISS (distance L2 sur vecteurs normalisés)
            distances, indices = self.index.search(query_vector, top_k)

            # Récupération et formatage des résultats
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # Vérifier que l'index est valide
                if idx == -1 or idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                
                # CORRECTION: Calcul de la vraie similarité cosinus
                # Pour vecteurs normalisés: distance_L2 = sqrt(2 - 2*cos_sim)
                # Donc: cos_sim = 1 - (distance_L2^2 / 2)
                cosine_similarity = 1.0 - (distance ** 2) / 2.0
                
                # Filtrage par score (similarité cosinus min)
                if cosine_similarity < min_score:
                    continue
                
                result = {
                    "document_id": doc.get("document_id", "unknown"),
                    "chunk_id": doc.get("chunk_id", "unknown"),
                    "text": doc.get("text", ""),
                    "score": float(cosine_similarity),  # Vraie similarité cosinus [-1, 1]
                    "distance": float(distance),  # Distance L2 originale pour debug
                    "word_count": doc.get("word_count", 0),
                    "char_count": doc.get("char_count", 0),
                    "embedding_norm": doc.get("embedding_norm", 0.0),
                    "source_file": doc.get("source_file", "unknown")
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            return []

    def get_index_stats(self) -> Dict[str, Union[int, str]]:
        """Retourne des statistiques sur l'index FAISS"""
        try:
            # Compter les documents uniques
            doc_ids = set(doc.get("document_id", "unknown") for doc in self.documents)
            
            stats = {
                "total_vectors": self.index.ntotal,
                "vector_dimension": self.index.d,
                "index_type": type(self.index).__name__,
                "unique_documents": len(doc_ids),
                "total_chunks": len(self.documents),
                "avg_chunks_per_doc": len(self.documents) / len(doc_ids) if doc_ids else 0,
                "model_device": str(self.model.device),
                "metadata_format": "new" if self.metadata_info else "legacy",
                "similarity_metric": "cosine"  # Ajout pour clarifier
            }
            
            return stats
            
        except Exception:
            return {
                "total_vectors": 0,
                "vector_dimension": 0,
                "index_type": "unknown",
                "unique_documents": 0,
                "total_chunks": 0,
                "avg_chunks_per_doc": 0,
                "model_device": "unknown",
                "metadata_format": "unknown",
                "similarity_metric": "cosine"
            }

    def debug_search(self, query: str, top_k: int = 3) -> Dict:
        """Version debug de la recherche avec informations détaillées"""
        # Embedding de la requête pour analyse
        query_vector = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        query_norm = float(np.linalg.norm(query_vector))
        
        results = self.search(query, top_k)
        
        debug_info = {
            "query": query,
            "query_embedding_norm": query_norm,
            "query_embedding_dim": len(query_vector[0]),
            "num_results": len(results),
            "index_stats": self.get_index_stats(),
            "results": results,
            "similarity_type": "cosine"  # Clarification pour debug
        }
        
        return debug_info

    def find_similar_chunks(self, document_id: str, chunk_id: str, top_k: int = 5) -> List[Dict]:
        """Trouve des chunks similaires à un chunk donné"""
        try:
            # Trouver le chunk source
            source_doc = None
            source_idx = None
            
            for i, doc in enumerate(self.documents):
                if (doc.get("document_id") == document_id and 
                    str(doc.get("chunk_id")) == str(chunk_id)):
                    source_doc = doc
                    source_idx = i
                    break
            
            if source_doc is None:
                return []
            
            # Utiliser son embedding pour la recherche
            # Créer un vecteur query à partir de l'embedding existant
            query_vector = np.array([source_doc.get("embedding", [])], dtype=np.float32)
            if query_vector.size == 0:
                return []
            
            # Recherche
            distances, indices = self.index.search(query_vector, top_k + 1)  # +1 pour exclure lui-même
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == source_idx:  # Exclure le chunk source
                    continue
                if idx == -1 or idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                
                # CORRECTION: Vraie similarité cosinus ici aussi
                cosine_similarity = 1.0 - (distance ** 2) / 2.0
                
                result = {
                    "document_id": doc.get("document_id", "unknown"),
                    "chunk_id": doc.get("chunk_id", "unknown"),
                    "text": doc.get("text", ""),
                    "score": float(cosine_similarity),
                    "distance": float(distance)
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception:
            return []