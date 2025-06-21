from pathlib import Path
from typing import List, Dict, Union
import json

from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import open_dir, exists_in
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
from whoosh import scoring
from whoosh.query import Or, Term, Every  # ← AJOUTER Every ici


class BM25Retriever:
    def __init__(self, index_dir: Union[str, Path]):
        """
        Initialise le retriever BM25
        
        Args:
            index_dir: Répertoire contenant l'index Whoosh créé par le script 04
        """
        self.index_dir = Path(index_dir)
        
        # Schéma enrichi compatible avec le nouveau format
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

        # Chargement de l'index existant
        if not exists_in(self.index_dir):
            raise FileNotFoundError(
                f"Index BM25 non trouvé dans {self.index_dir}. "
                f"Exécutez d'abord: poetry run python scripts/04_index_bm25.py"
            )
        
        self.index = open_dir(self.index_dir)

    def _get_index_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur l'index - VERSION CORRIGÉE"""
        try:
            with self.index.searcher() as searcher:
                total_docs = searcher.doc_count()
                
                if total_docs == 0:
                    return {"total_chunks": 0, "unique_documents": 0, "avg_chunks_per_doc": 0}
                
                # MÉTHODE CORRIGÉE : utiliser Every() pour récupérer tous les documents
                try:
                    all_docs_query = Every()
                    results = searcher.search(all_docs_query, limit=None)  # Pas de limite
                    
                    # Compter les documents uniques
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
                    # Fallback : estimation simple si Every() ne marche pas
                    estimated_docs = max(1, total_docs // 10)  # ~10 chunks par doc
                    return {
                        "total_chunks": total_docs,
                        "unique_documents": estimated_docs,
                        "avg_chunks_per_doc": total_docs / estimated_docs
                    }
                    
        except Exception:
            return {"total_chunks": 0, "unique_documents": 0, "avg_chunks_per_doc": 0}

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Union[str, float]]]:
        """
        Recherche dans l'index BM25
        
        Args:
            query: Requête de recherche
            top_k: Nombre maximum de résultats
            min_score: Score minimum pour filtrer les résultats
            
        Returns:
            Liste des chunks trouvés avec métadonnées
        """
        if not query.strip():
            return []
        
        try:
            with self.index.searcher(weighting=scoring.BM25F()) as searcher:
                parser = QueryParser("content", schema=self.schema)
                
                # Nettoyage de la requête pour éviter les erreurs de parsing
                cleaned_query = self._clean_query(query)
                
                try:
                    parsed_query = parser.parse(cleaned_query)
                except Exception:
                    # Fallback: recherche simple par termes si parsing échoue
                    words = query.split()
                    terms = [Term("content", word.lower()) for word in words if word.strip()]
                    if not terms:
                        return []
                    parsed_query = Or(terms)
                
                # Recherche avec limite
                results = searcher.search(parsed_query, limit=max(top_k * 2, 20))
                
                # Formatage des résultats avec métadonnées enrichies
                formatted_results = []
                for hit in results:
                    score = float(hit.score)
                    
                    # Filtrage par score minimum
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
                    
                    # Limite après filtrage
                    if len(formatted_results) >= top_k:
                        break
                
                return formatted_results
                
        except Exception:
            return []

    def _clean_query(self, query: str) -> str:
        """Nettoie la requête pour éviter les erreurs de parsing Whoosh"""
        # Remplacer les caractères problématiques
        replacements = {
            '"': '',  # Guillemets problématiques
            "'": '',  # Apostrophes
            '(': '',  # Parenthèses
            ')': '',
            '[': '',  # Crochets
            ']': '',
            '{': '',  # Accolades
            '}': '',
            ':': ' ',  # Deux points peuvent causer des problèmes
            ';': ' ',
            '!': '',   # Points d'exclamation
            '?': '',   # Points d'interrogation
            '*': '',   # Wildcards
            '+': ' ',  # Opérateurs
            '-': ' ',
            '/': ' ',  # Slashes
            '\\': ' ',
            '|': ' ',  # Pipes
            '&': ' ',  # Ampersands
        }
        
        cleaned = query
        for char, replacement in replacements.items():
            cleaned = cleaned.replace(char, replacement)
        
        # Supprimer les espaces multiples
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()

    def get_document_stats(self) -> Dict[str, int]:
        """Retourne des statistiques sur l'index"""
        return self._get_index_stats()

    def debug_search(self, query: str, top_k: int = 3) -> Dict:
        """Version debug de la recherche avec informations détaillées"""
        results = self.search(query, top_k)
        
        debug_info = {
            "original_query": query,
            "cleaned_query": self._clean_query(query),
            "num_results": len(results),
            "index_stats": self.get_document_stats(),
            "results": results
        }
        
        return debug_info