"""
Semantic Search: FAISS-Based Vector Retrieval System

This module provides comprehensive semantic search capabilities using FAISS indexing
and SentenceTransformer embeddings. It implements cosine similarity-based retrieval,
robust metadata handling, and comprehensive result formatting for the RAG diagnosis system.

Key components:
- FAISS vector search with cosine similarity computation and normalization
- SentenceTransformer embedding generation with GPU acceleration support
- Multi-format metadata compatibility for legacy and new index structures
- Comprehensive similarity analysis and chunk comparison capabilities
- Debug modes for embedding analysis and performance monitoring

Dependencies: faiss, sentence-transformers, torch, numpy, pickle
Usage: Import FAISSRetriever for semantic search operations with vector similarity analysis
"""

import pickle
from pathlib import Path
from typing import List, Dict, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    """
    FAISS-based semantic retriever with cosine similarity and comprehensive metadata support.
    
    Implements sophisticated vector search functionality using FAISS indexing with
    SentenceTransformer embeddings, supporting multiple metadata formats and providing
    accurate cosine similarity calculations for optimal semantic matching.
    """
    
    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize FAISS retriever with index loading and model setup.
        
        Sets up the semantic retriever with FAISS index loading, metadata parsing,
        and SentenceTransformer model initialization with dimension validation.
        
        Args:
            index_path (Union[str, Path]): Path to FAISS index file (.faiss)
            metadata_path (Union[str, Path]): Path to metadata file (.pkl)
            embedding_model_name (str): Name of SentenceTransformer model
            
        Raises:
            FileNotFoundError: If index or metadata files are not found
            ValueError: If metadata format is unrecognized or dimensions are incompatible
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Existence verification
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}. "
                f"Run first: poetry run python scripts/05_create_faiss_index.py"
            )
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {self.metadata_path}."
            )

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load metadata with support for old/new formats
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
        
        # Metadata format detection
        if isinstance(data, dict) and "documents" in data:
            # New format with metadata
            self.documents = data["documents"]
            self.ids = data.get("ids", [])
            self.metadata_info = data.get("metadata", {})
        elif isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
            # Intermediate format
            self.documents = data["documents"]
            self.ids = [f"{doc['document_id']}|{doc['chunk_id']}" for doc in self.documents]
            self.metadata_info = {}
        elif isinstance(data, list):
            # Legacy format - direct list
            self.documents = data
            self.ids = [f"{doc['document_id']}|{doc['chunk_id']}" for doc in data]
            self.metadata_info = {}
        else:
            raise ValueError("Unrecognized metadata format")

        # Validation
        if len(self.documents) != self.index.ntotal:
            raise ValueError(
                f"Inconsistency: {len(self.documents)} documents vs {self.index.ntotal} vectors in index"
            )

        # Load SentenceTransformer model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(embedding_model_name, device=device)
        
        # Dimension compatibility verification
        expected_dim = self.model.get_sentence_embedding_dimension()
        if self.index.d != expected_dim:
            raise ValueError(
                f"Incompatible dimension: model={expected_dim}, index={self.index.d}"
            )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Perform semantic search in FAISS index with cosine similarity.
        
        Executes vector-based semantic search using normalized embeddings and
        accurate cosine similarity calculation with comprehensive result formatting.
        
        Args:
            query (str): Search query string
            top_k (int): Maximum number of results to return
            min_score (float): Minimum score threshold (cosine similarity minimum)
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of found chunks with comprehensive metadata
        """
        if not query.strip():
            return []
        
        try:
            # Generate query embedding (normalized for cosine similarity)
            query_vector = self.model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
            query_vector = query_vector.astype("float32")

            # Search in FAISS index (L2 distance on normalized vectors)
            distances, indices = self.index.search(query_vector, top_k)

            # Result retrieval and formatting
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                # Verify index is valid
                if idx == -1 or idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                
                # Correction: Calculate true cosine similarity
                # For normalized vectors: distance_L2 = sqrt(2 - 2*cos_sim)
                # Therefore: cos_sim = 1 - (distance_L2^2 / 2)
                cosine_similarity = 1.0 - (distance ** 2) / 2.0
                
                # Filtering by score (minimum cosine similarity)
                if cosine_similarity < min_score:
                    continue
                
                result = {
                    "document_id": doc.get("document_id", "unknown"),
                    "chunk_id": doc.get("chunk_id", "unknown"),
                    "text": doc.get("text", ""),
                    "score": float(cosine_similarity),  # True cosine similarity [-1, 1]
                    "distance": float(distance),  # Original L2 distance for debug
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
        """
        Return comprehensive FAISS index statistics for monitoring.
        
        Provides detailed index information including vector counts, dimensions,
        document statistics, and metadata format for system health assessment.
        
        Returns:
            Dict[str, Union[int, str]]: Complete index statistics and configuration
        """
        try:
            # Count unique documents
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
                "similarity_metric": "cosine"
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
        """
        Debug version of search with detailed embedding and similarity analysis.
        
        Provides comprehensive search debugging including embedding analysis,
        similarity calculations, and index statistics for troubleshooting
        and performance optimization.
        
        Args:
            query (str): Search query for debugging
            top_k (int): Number of results for analysis
            
        Returns:
            Dict: Complete debug information including embedding details and results
        """
        # Query embedding for analysis
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
            "similarity_type": "cosine"
        }
        
        return debug_info

    def find_similar_chunks(self, document_id: str, chunk_id: str, top_k: int = 5) -> List[Dict]:
        """
        Find chunks similar to a given chunk using vector similarity.
        
        Performs similarity search using the embedding of a specified chunk
        to find related content with accurate cosine similarity calculation.
        
        Args:
            document_id (str): Source document identifier
            chunk_id (str): Source chunk identifier
            top_k (int): Number of similar chunks to return
            
        Returns:
            List[Dict]: List of similar chunks with similarity scores
        """
        try:
            # Find source chunk
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
            
            # Use its embedding for search
            # Create query vector from existing embedding
            query_vector = np.array([source_doc.get("embedding", [])], dtype=np.float32)
            if query_vector.size == 0:
                return []
            
            # Search
            distances, indices = self.index.search(query_vector, top_k + 1)  # +1 to exclude itself
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == source_idx:  # Exclude source chunk
                    continue
                if idx == -1 or idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                
                # Correction: True cosine similarity here too
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