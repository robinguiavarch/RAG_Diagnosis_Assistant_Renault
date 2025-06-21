"""
Module de préprocessing des requêtes avec LLM
Version simplifiée pour comparateur RAG
"""

from .llm_client import LLMClient, create_llm_client
from .response_parser import ProcessedQuery, TechnicalTerms, EquipmentInfo, parse_llm_response
from .unified_query_processor import UnifiedQueryProcessor, create_query_processor, process_single_query
from .enhanced_retrieval_engine import EnhancedRetrievalEngine, RetrievalResult, create_enhanced_retrieval_engine

# Exports principaux
__all__ = [
    "LLMClient",
    "ProcessedQuery", 
    "UnifiedQueryProcessor",
    "EnhancedRetrievalEngine",
    "create_query_processor",
    "process_single_query",
    "create_enhanced_retrieval_engine"
]

# Fonction utilitaire principale
def quick_setup() -> UnifiedQueryProcessor:
    """Configuration rapide pour usage immédiat"""
    return create_query_processor()