"""
Lexical Search: BM25-Based Document Retrieval System

This module provides comprehensive lexical search capabilities using the BM25 algorithm
through Whoosh indexing. It implements robust query processing, index management,
and result formatting with comprehensive metadata support for the RAG diagnosis system.

Key components:
- BM25 retrieval with configurable scoring parameters and result filtering
- Robust query cleaning and parsing with fallback mechanisms
- Comprehensive index statistics and document counting capabilities
- Enhanced result formatting with quality metrics and source attribution
- Debug modes for search analysis and performance monitoring

Dependencies: whoosh, pathlib, json
Usage: Import BM25Retriever for lexical search operations with enriched metadata support
"""

from pathlib import Path
from typing import List, Dict, Union
import json

from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import open_dir, exists_in
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
from whoosh import scoring
from whoosh.query import Or, Term, Every


class BM25Retriever:
    """
    BM25-based lexical retriever with robust query processing and metadata support.
    
    Implements comprehensive BM25 search functionality with enhanced query cleaning,
    result formatting, and index management capabilities for optimal performance
    in the RAG diagnosis system.
    """
    
    def __init__(self, index_dir: Union[str, Path]):
        """
        Initialize BM25 retriever with index loading and schema validation.
        
        Sets up the retriever with enriched schema support and validates
        the existence of the required Whoosh index directory.
        
        Args:
            index_dir (Union[str, Path]): Directory containing Whoosh index created by indexing script
            
        Raises:
            FileNotFoundError: If BM25 index is not found in specified directory
        """
        self.index_dir = Path(index_dir)
        
        # Enriched schema compatible with new format
        self.schema = Schema(
            document_id=ID(stored=True),
            chunk_id=ID(stored=True),
            content=TEXT(analyzer=StandardAnalyzer(), stored=True),
            word_count=NUMERIC(stored=True),
            char_count=NUMERIC(stored=True),
            quality_score=NUMERIC(stored=True),
            source_file=ID(stored=True),
            chunking_method=ID(stored=True)
        )

        # Load existing index
        if not exists_in(self.index_dir):
            raise FileNotFoundError(
                f"BM25 index not found in {self.index_dir}. "
                f"Run first: poetry run python scripts/04_index_bm25.py"
            )
        
        self.index = open_dir(self.index_dir)

    def _get_index_stats(self) -> Dict[str, int]:
        """
        Return comprehensive index statistics with corrected document counting.
        
        Computes accurate statistics about the index including total chunks,
        unique documents, and average chunks per document using robust
        query mechanisms with fallback support.
        
        Returns:
            Dict[str, int]: Index statistics including counts and averages
        """
        try:
            with self.index.searcher() as searcher:
                total_docs = searcher.doc_count()
                
                if total_docs == 0:
                    return {"total_chunks": 0, "unique_documents": 0, "avg_chunks_per_doc": 0}
                
                # Corrected method: use Every() to retrieve all documents
                try:
                    all_docs_query = Every()
                    results = searcher.search(all_docs_query, limit=None)  # No limit
                    
                    # Count unique documents
                    doc_ids = set()
                    for hit in results:
                        doc_id = hit.get("document_id", "unknown")
                        if doc_id != "unknown":
                            doc_ids.add(doc_id)
                    
                    return {
                        "total_chunks": total_docs,
                        "unique_documents": len(doc_ids),
                        "avg_chunks_per_doc": total_docs / len(doc_ids) if doc_ids else 0
                    }
                    
                except Exception:
                    # Fallback: simple estimation if Every() doesn't work
                    estimated_docs = max(1, total_docs // 10)  # ~10 chunks per doc
                    return {
                        "total_chunks": total_docs,
                        "unique_documents": estimated_docs,
                        "avg_chunks_per_doc": total_docs / estimated_docs
                    }
                    
        except Exception:
            return {"total_chunks": 0, "unique_documents": 0, "avg_chunks_per_doc": 0}

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Perform BM25 search with comprehensive result formatting and filtering.
        
        Executes lexical search using BM25 scoring with robust query processing,
        result filtering, and enriched metadata extraction for optimal performance.
        
        Args:
            query (str): Search query string
            top_k (int): Maximum number of results to return
            min_score (float): Minimum score threshold for result filtering
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of found chunks with comprehensive metadata
        """
        if not query.strip():
            return []
        
        try:
            with self.index.searcher(weighting=scoring.BM25F()) as searcher:
                parser = QueryParser("content", schema=self.schema)
                
                # Query cleaning to avoid parsing errors
                cleaned_query = self._clean_query(query)
                
                try:
                    parsed_query = parser.parse(cleaned_query)
                except Exception:
                    # Fallback: simple term search if parsing fails
                    words = query.split()
                    terms = [Term("content", word.lower()) for word in words if word.strip()]
                    if not terms:
                        return []
                    parsed_query = Or(terms)
                
                # Search with limit
                results = searcher.search(parsed_query, limit=max(top_k * 2, 20))
                
                # Result formatting with enriched metadata
                formatted_results = []
                for hit in results:
                    score = float(hit.score)
                    
                    # Minimum score filtering
                    if score < min_score:
                        continue
                    
                    result = {
                        "document_id": hit.get("document_id", "unknown"),
                        "chunk_id": hit.get("chunk_id", "unknown"),
                        "text": hit.get("content", ""),
                        "score": score,
                        "word_count": hit.get("word_count", 0),
                        "char_count": hit.get("char_count", 0),
                        "quality_score": hit.get("quality_score", 0.0),
                        "source_file": hit.get("source_file", "unknown"),
                        "chunking_method": hit.get("chunking_method", "unknown")
                    }
                    formatted_results.append(result)
                    
                    # Limit after filtering
                    if len(formatted_results) >= top_k:
                        break
                
                return formatted_results
                
        except Exception:
            return []

    def _clean_query(self, query: str) -> str:
        """
        Clean query string to prevent Whoosh parsing errors.
        
        Removes or replaces problematic characters that could cause parsing
        failures in Whoosh query processing, ensuring robust search operation.
        
        Args:
            query (str): Raw query string to clean
            
        Returns:
            str: Cleaned query string safe for Whoosh parsing
        """
        # Replace problematic characters
        replacements = {
            '"': '',  # Problematic quotes
            "'": '',  # Apostrophes
            '(': '',  # Parentheses
            ')': '',
            '[': '',  # Brackets
            ']': '',
            '{': '',  # Braces
            '}': '',
            ':': ' ',  # Colons can cause problems
            ';': ' ',
            '!': '',   # Exclamation marks
            '?': '',   # Question marks
            '*': '',   # Wildcards
            '+': ' ',  # Operators
            '-': ' ',
            '/': ' ',  # Slashes
            '\\': ' ',
            '|': ' ',  # Pipes
            '&': ' ',  # Ampersands
        }
        
        cleaned = query
        for char, replacement in replacements.items():
            cleaned = cleaned.replace(char, replacement)
        
        # Remove multiple spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()

    def get_document_stats(self) -> Dict[str, int]:
        """
        Return comprehensive document statistics for index monitoring.
        
        Provides access to index statistics for performance monitoring,
        capacity planning, and system health assessment.
        
        Returns:
            Dict[str, int]: Complete index statistics and metrics
        """
        return self._get_index_stats()

    def debug_search(self, query: str, top_k: int = 3) -> Dict:
        """
        Debug version of search with detailed information and analysis.
        
        Provides comprehensive search debugging including query processing
        details, index statistics, and result analysis for troubleshooting
        and performance optimization.
        
        Args:
            query (str): Search query for debugging
            top_k (int): Number of results for analysis
            
        Returns:
            Dict: Complete debug information including query processing and results
        """
        results = self.search(query, top_k)
        
        debug_info = {
            "original_query": query,
            "cleaned_query": self._clean_query(query),
            "num_results": len(results),
            "index_stats": self.get_document_stats(),
            "results": results
        }
        
        return debug_info