"""
Enhanced Retrieval Engine: Multi-Variant Search with Deduplication

This module implements an advanced retrieval system that processes multiple query variants
and combines results from different search methods. It integrates lexical search, semantic
search, hybrid fusion, and knowledge graph retrieval to provide comprehensive document
retrieval with automatic deduplication and reranking capabilities.

Key components:
- Multi-variant query processing with candidate gathering from all variants
- Integration with existing BM25, FAISS, and hybrid fusion modules
- Simple content-based deduplication using hash comparison
- Final ranking using CrossEncoder reranking with primary query
- Knowledge graph triplet extraction for enhanced context

Dependencies: hashlib, time, dataclasses, existing retrieval modules
Usage: Import EnhancedRetrievalEngine for comprehensive multi-variant document retrieval
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

# Project imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .response_parser import ProcessedQuery
from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever
from core.retrieval_engine.hybrid_fusion import fuse_results
from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
from core.retrieval_graph.dense_kg_querier import get_structured_context


@dataclass
class RetrievalResult:
    """Enhanced retrieval result container with processing metrics"""
    chunks: List[Dict[str, Any]]
    triplets: List[Dict[str, Any]]
    processing_time: float
    variants_used: int


class EnhancedRetrievalEngine:
    """
    Enhanced retrieval engine with multi-query variant support.
    
    Integrates existing retrieval modules (BM25, FAISS, fusion, reranking)
    to provide comprehensive search capabilities with automatic deduplication
    and intelligent ranking across multiple query formulations.
    """
    
    def __init__(self, 
                 bm25_retriever: BM25Retriever,
                 faiss_retriever: FAISSRetriever,
                 reranker: CrossEncoderReranker,
                 pool_size: int = 15,
                 final_top_k: int = 5):
        """
        Initialize enhanced retrieval engine with configured retrievers.
        
        Sets up the multi-variant retrieval system with specified retrievers
        and ranking parameters for optimal search performance.
        
        Args:
            bm25_retriever (BM25Retriever): Lexical search retriever instance
            faiss_retriever (FAISSRetriever): Semantic search retriever instance
            reranker (CrossEncoderReranker): Reranking module for final scoring
            pool_size (int): Size of candidate pool for intermediate results
            final_top_k (int): Number of final results to return
        """
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.reranker = reranker
        self.pool_size = pool_size
        self.final_top_k = final_top_k
    
    def search_with_variants(self, processed_query: ProcessedQuery) -> RetrievalResult:
        """
        Perform search using multiple query variants with comprehensive retrieval.
        
        Orchestrates the complete retrieval pipeline including candidate gathering,
        deduplication, fusion, reranking, and knowledge graph enhancement.
        
        Args:
            processed_query (ProcessedQuery): LLM-processed query with variants
            
        Returns:
            RetrievalResult: Comprehensive retrieval results with metrics
        """
        start_time = time.time()
        
        # Step 1: Gather candidates from all variants
        chunk_pool = self._gather_chunk_candidates(processed_query)
        
        # Step 2: Simple deduplication
        unique_chunks = self._deduplicate_chunks(chunk_pool)
        
        # Step 3: Fusion and reranking with primary query
        final_chunks = self._final_ranking(processed_query.get_primary_query(), unique_chunks)
        
        # Step 4: Knowledge graph search
        triplets = self._get_kg_triplets(processed_query.get_primary_query())
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            chunks=final_chunks,
            triplets=triplets,
            processing_time=processing_time,
            variants_used=len(processed_query.query_variants)
        )
    
    def _gather_chunk_candidates(self, processed_query: ProcessedQuery) -> List[Dict]:
        """
        Gather candidates from all retrievers and query variants.
        
        Collects search results from both BM25 and FAISS retrievers across
        all available query variants to maximize recall.
        
        Args:
            processed_query (ProcessedQuery): Processed query with variants
            
        Returns:
            List[Dict]: Combined pool of candidate chunks from all sources
        """
        chunk_pool = []
        
        # Queries to process (primary + variants, max 3)
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
        """
        Simple deduplication based on content hash.
        
        Removes duplicate chunks using MD5 hash comparison of text content
        to ensure unique results in the candidate pool.
        
        Args:
            chunk_pool (List[Dict]): Pool of candidate chunks with potential duplicates
            
        Returns:
            List[Dict]: Deduplicated list of unique chunks
        """
        if not chunk_pool:
            return []
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunk_pool:
            content = chunk.get("text", "").strip()
            if not content:
                continue
            
            # Content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _final_ranking(self, primary_query: str, unique_chunks: List[Dict]) -> List[Dict]:
        """
        Final fusion and reranking with primary query.
        
        Performs hybrid fusion of BM25 and FAISS results followed by
        CrossEncoder reranking to produce final ranked results.
        
        Args:
            primary_query (str): Primary query for final ranking
            unique_chunks (List[Dict]): Deduplicated candidate chunks
            
        Returns:
            List[Dict]: Final ranked chunks limited to top_k results
        """
        if not unique_chunks:
            return []
        
        # Separate by source for fusion
        bm25_chunks = [c for c in unique_chunks if "BM25" in c.get("source", "")]
        faiss_chunks = [c for c in unique_chunks if "FAISS" in c.get("source", "")]
        
        # Fusion if both types available
        if bm25_chunks and faiss_chunks:
            fused_chunks = fuse_results(bm25_chunks, faiss_chunks, top_k=self.pool_size)
        else:
            fused_chunks = bm25_chunks + faiss_chunks
        
        # Reranking with CrossEncoder
        if self.reranker and fused_chunks:
            try:
                return self.reranker.rerank(
                    query=primary_query,
                    candidates=fused_chunks,
                    top_k=self.final_top_k
                )
            except Exception:
                pass
        
        # Fallback: score-based sorting
        fused_chunks.sort(key=lambda x: x.get("fused_score", x.get("score", 0)), reverse=True)
        return fused_chunks[:self.final_top_k]
    
    def _get_kg_triplets(self, query: str) -> List[Dict]:
        """
        Retrieve triplets from knowledge graph.
        
        Extracts relevant knowledge graph triplets for the query to provide
        additional structured context for response generation.
        
        Args:
            query (str): Search query for knowledge graph lookup
            
        Returns:
            List[Dict]: List of relevant knowledge graph triplets
        """
        try:
            kg_context = get_structured_context(query, format_type="json", max_triplets=3)
            
            if kg_context and "No relevant" not in kg_context:
                import json
                triplets = json.loads(kg_context)
                return triplets if isinstance(triplets, list) else []
        except Exception:
            pass
        
        return []


def create_enhanced_retrieval_engine(bm25_retriever: BM25Retriever,
                                   faiss_retriever: FAISSRetriever,
                                   reranker: CrossEncoderReranker) -> EnhancedRetrievalEngine:
    """
    Create enhanced retrieval engine instance.
    
    Factory function for convenient instantiation of enhanced retrieval engine
    with standard configuration and all required components.
    
    Args:
        bm25_retriever (BM25Retriever): Configured BM25 retriever
        faiss_retriever (FAISSRetriever): Configured FAISS retriever
        reranker (CrossEncoderReranker): Configured reranker
        
    Returns:
        EnhancedRetrievalEngine: Fully configured enhanced retrieval engine
    """
    return EnhancedRetrievalEngine(
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        reranker=reranker
    )


if __name__ == "__main__":
    print("Enhanced Retrieval Engine Test")
    print("Simplified module ready for use")