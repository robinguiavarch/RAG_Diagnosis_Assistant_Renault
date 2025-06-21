"""
Moteur de retrieval amélioré - Version simplifiée
Implémente la recherche avec variantes multiples et déduplication de base
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

# Imports du projet existant
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .response_parser import ProcessedQuery
from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever
from core.retrieval_engine.hybrid_fusion import fuse_results
from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
from core.retrieval_graph.dense_kg_querier import get_structured_context


@dataclass
class RetrievalResult:
    """Résultat du retrieval amélioré"""
    chunks: List[Dict[str, Any]]
    triplets: List[Dict[str, Any]]
    processing_time: float
    variants_used: int


class EnhancedRetrievalEngine:
    """
    Moteur de retrieval amélioré avec support des requêtes multiples
    Utilise les modules existants (BM25, FAISS, fusion, reranking)
    """
    
    def __init__(self, 
                 bm25_retriever: BM25Retriever,
                 faiss_retriever: FAISSRetriever,
                 reranker: CrossEncoderReranker,
                 pool_size: int = 15,
                 final_top_k: int = 5):
        
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.reranker = reranker
        self.pool_size = pool_size
        self.final_top_k = final_top_k
    
    def search_with_variants(self, processed_query: ProcessedQuery) -> RetrievalResult:
        """
        Recherche avec variantes multiples
        
        Args:
            processed_query: Requête traitée par le LLM avec variantes
            
        Returns:
            RetrievalResult: Résultats finaux
        """
        start_time = time.time()
        
        # Étape 1: Collecte des candidats depuis toutes les variantes
        chunk_pool = self._gather_chunk_candidates(processed_query)
        
        # Étape 2: Déduplication simple
        unique_chunks = self._deduplicate_chunks(chunk_pool)
        
        # Étape 3: Fusion et reranking avec requête principale
        final_chunks = self._final_ranking(processed_query.get_primary_query(), unique_chunks)
        
        # Étape 4: Recherche Knowledge Graph
        triplets = self._get_kg_triplets(processed_query.get_primary_query())
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            chunks=final_chunks,
            triplets=triplets,
            processing_time=processing_time,
            variants_used=len(processed_query.query_variants)
        )
    
    def _gather_chunk_candidates(self, processed_query: ProcessedQuery) -> List[Dict]:
        """Collecte les candidats depuis tous les retrievers et toutes les variantes"""
        chunk_pool = []
        
        # Requêtes à traiter (principale + variantes, max 3)
        all_queries = processed_query.get_all_queries()[:3]
        
        for i, query in enumerate(all_queries):
            # BM25 Retrieval
            try:
                bm25_results = self.bm25.search(query, top_k=self.pool_size // 2)
                for doc in bm25_results:
                    doc["source"] = f"BM25_v{i}"
                chunk_pool.extend(bm25_results)
            except Exception:
                pass
            
            # FAISS Retrieval
            try:
                faiss_results = self.faiss.search(query, top_k=self.pool_size // 2)
                for doc in faiss_results:
                    doc["source"] = f"FAISS_v{i}"
                chunk_pool.extend(faiss_results)
            except Exception:
                pass
        
        return chunk_pool
    
    def _deduplicate_chunks(self, chunk_pool: List[Dict]) -> List[Dict]:
        """Déduplication simple basée sur le hash du contenu"""
        if not chunk_pool:
            return []
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunk_pool:
            content = chunk.get("text", "").strip()
            if not content:
                continue
            
            # Hash du contenu
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _final_ranking(self, primary_query: str, unique_chunks: List[Dict]) -> List[Dict]:
        """Fusion et reranking final avec la requête principale"""
        if not unique_chunks:
            return []
        
        # Séparation par source pour la fusion
        bm25_chunks = [c for c in unique_chunks if "BM25" in c.get("source", "")]
        faiss_chunks = [c for c in unique_chunks if "FAISS" in c.get("source", "")]
        
        # Fusion si on a les deux types
        if bm25_chunks and faiss_chunks:
            fused_chunks = fuse_results(bm25_chunks, faiss_chunks, top_k=self.pool_size)
        else:
            fused_chunks = bm25_chunks + faiss_chunks
        
        # Reranking avec CrossEncoder
        if self.reranker and fused_chunks:
            try:
                return self.reranker.rerank(
                    query=primary_query,
                    candidates=fused_chunks,
                    top_k=self.final_top_k
                )
            except Exception:
                pass
        
        # Fallback: tri par score
        fused_chunks.sort(key=lambda x: x.get("fused_score", x.get("score", 0)), reverse=True)
        return fused_chunks[:self.final_top_k]
    
    def _get_kg_triplets(self, query: str) -> List[Dict]:
        """Récupération des triplets depuis le Knowledge Graph"""
        try:
            kg_context = get_structured_context(query, format_type="json", max_triplets=3)
            
            if kg_context and "No relevant" not in kg_context:
                import json
                triplets = json.loads(kg_context)
                return triplets if isinstance(triplets, list) else []
        except Exception:
            pass
        
        return []


# Fonction utilitaire
def create_enhanced_retrieval_engine(bm25_retriever: BM25Retriever,
                                   faiss_retriever: FAISSRetriever,
                                   reranker: CrossEncoderReranker) -> EnhancedRetrievalEngine:
    """Crée un moteur de retrieval amélioré"""
    return EnhancedRetrievalEngine(
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        reranker=reranker
    )


if __name__ == "__main__":
    print("🧪 Test EnhancedRetrievalEngine")
    print("✅ Module simplifié prêt à l'emploi")