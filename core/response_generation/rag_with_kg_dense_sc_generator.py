"""
RAG with Dense Symptom-Cause Knowledge Graph Generator: Enhanced RAG with S&C Integration

This module provides advanced RAG generation capabilities enhanced with dense symptom-cause
knowledge graph integration. It implements multi-query processing, intelligent context
evaluation, adaptive prompt strategies, and sophisticated relevance assessment using
CrossEncoder scores with comprehensive fallback mechanisms specific to symptom-cause analysis.

Key components:
- Dense symptom-cause knowledge graph integration with multi-query and equipment filtering
- CrossEncoder score normalization and document relevance evaluation
- Adaptive prompt strategies based on context availability and quality
- Strict context management with configurable chunk and token limitations
- External prompt template loading with multi-strategy support for S&C enriched structure

Dependencies: openai, yaml, pathlib, math, dense_sc_kg_querier
Usage: Import OpenAIGeneratorKGDenseSC for enhanced RAG with dense S&C knowledge graph integration
"""

import os
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import yaml
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import enhanced multi-query functions for dense S&C KG
from core.retrieval_graph.dense_sc_kg_querier import (
    get_structured_context_dense_sc_with_multi_query,
    get_structured_context_dense_sc_with_equipment_filter,
    get_structured_context_dense_sc
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_cross_encoder_score(raw_score: float) -> float:
    """
    Normalize CrossEncoder scores to 0-1 range using sigmoid function.
    
    Args:
        raw_score (float): Raw CrossEncoder score
    
    Returns:
        float: Normalized score between 0 and 1
    """
    try:
        normalized = 1.0 / (1.0 + math.exp(-raw_score))
        return float(normalized)
    except (OverflowError, ZeroDivisionError):
        if raw_score > 0:
            return 1.0
        else:
            return 0.0

def evaluate_document_relevance(reranked_docs: List[Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    """
    Evaluate document relevance based on normalized CrossEncoder scores.
    
    Performs comprehensive relevance assessment using normalized scoring with
    configurable thresholds for quality control in dense S&C context.
    
    Args:
        reranked_docs (List[Dict[str, Any]]): Documents with CrossEncoder scores
        threshold (float): Relevance threshold (after normalization, 0-1 range)
    
    Returns:
        Dict[str, Any]: Comprehensive relevance statistics and assessment
    """
    if not reranked_docs:
        return {
            "is_relevant": False,
            "relevant_count": 0,
            "total_count": 0,
            "max_score": 0.0,
            "avg_score": 0.0
        }
    
    normalized_scores = []
    relevant_count = 0
    
    for doc in reranked_docs:
        raw_score = doc.get('cross_encoder_score', 0.0)
        normalized_score = normalize_cross_encoder_score(raw_score)
        normalized_scores.append(normalized_score)
        doc['cross_encoder_score_normalized'] = normalized_score
        
        if normalized_score >= threshold:
            relevant_count += 1
    
    max_score = max(normalized_scores) if normalized_scores else 0.0
    avg_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    is_relevant = relevant_count >= 1
    
    stats = {
        "is_relevant": is_relevant,
        "relevant_count": relevant_count,
        "total_count": len(reranked_docs),
        "max_score": max_score,
        "avg_score": avg_score,
        "threshold_used": threshold
    }
    
    print(f"Dense S&C document relevance evaluation:")
    print(f"   Threshold used: {threshold}")
    print(f"   Relevant documents: {relevant_count}/{len(reranked_docs)}")
    print(f"   Max score (normalized): {max_score:.3f}")
    print(f"   Verdict: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
    
    return stats

class OpenAIGeneratorKGDenseSC:
    """
    Enhanced RAG generator with dense symptom-cause knowledge graph integration and multi-query support.
    
    Implements sophisticated RAG generation combining document retrieval with dense
    symptom-cause knowledge graph context, featuring adaptive prompt strategies,
    multi-query processing, and comprehensive relevance evaluation mechanisms.
    """
    
    def __init__(self, model: str = "gpt-4o", context_token_limit: int = 6000):
        """
        Initialize enhanced RAG generator with dense S&C knowledge graph capabilities.
        
        Sets up the generator with comprehensive configuration loading, prompt template
        management, and adaptive context processing for symptom-cause enriched structure.
        
        Args:
            model (str): OpenAI model name for generation
            context_token_limit (int): Maximum tokens allowed for context
        """
        self.model = model
        self.context_token_limit = context_token_limit

        # YAML parameter loading
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)

        gen_cfg = settings.get("generation", {})
        self.importance_context_rerank = gen_cfg.get("importance_context_rerank", 50)
        self.importance_context_graph = gen_cfg.get("importance_context_graph", 50)
        self.max_tokens = gen_cfg.get("max_new_tokens", 2000)
        
        # Load max_context_chunks from settings.yaml
        self.max_context_chunks = gen_cfg.get("max_context_chunks", 3)
        
        self.max_triplets = gen_cfg.get("top_k_triplets", 3)
        self.relevance_threshold = gen_cfg.get("seuil_pertinence", 0.7)
        
        # Load prompts for dense S&C KG
        self.prompts = self._load_prompt_templates()
        
        print(f"OpenAIGeneratorKGDenseSC initialized:")
        print(f"   Max chunks: {self.max_context_chunks}")
        print(f"   Token limit: {self.context_token_limit}")
        print(f"   Triplet limitation: {self.max_triplets}")
        print(f"   Relevance threshold: {self.relevance_threshold}")
        print(f"   Prompts loaded: {list(self.prompts.keys())}")

    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt templates for dense symptom-cause knowledge graph processing.
        
        Reads and parses dense S&C KG-specific prompt templates from external
        configuration file, supporting different context scenarios.
        
        Returns:
            Dict[str, str]: Dictionary of parsed prompt templates by strategy
            
        Raises:
            FileNotFoundError: If prompt file is missing
            ValueError: If required prompts are not found
        """
        prompt_path = Path("config/prompts/rag_with_kg_dense_s&c_prompt.txt")
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse prompts according to provided file format
        prompts = {}
        sections = content.split("# === PROMPT ")
        
        for section in sections[1:]:
            lines = section.split('\n')
            # Format: "KG_DENSE_SC_ONLY ===", "DOC_ONLY ===", "BOTH ==="
            prompt_header = lines[0].split(' ===')[0]
            prompt_content = '\n'.join(lines[1:]).strip()
            
            # Mapping to generator keys
            if prompt_header == "KG_DENSE_SC_ONLY":
                prompts["kg_only"] = prompt_content
            elif prompt_header == "DOC_ONLY":
                prompts["doc_only"] = prompt_content
            elif prompt_header == "BOTH":
                prompts["both"] = prompt_content
        
        if len(prompts) != 3:
            raise ValueError(f"Missing prompts in {prompt_path}. Found: {list(prompts.keys())}")
        
        return prompts

    def _estimate_tokens(self, text: str) -> int:
        """
        Simple token count estimation for budget management.
        
        Args:
            text (str): Text to estimate token count for
            
        Returns:
            int: Estimated token count
        """
        return int(len(text.split()) * 0.75)

    def select_passages_with_limits(self, passages: List[str]) -> tuple:
        """
        Strict passage selection with respect for maximum context chunks.
        
        Implements hierarchical selection prioritizing chunk count limits followed
        by token budget validation for optimal context utilization.
        
        Args:
            passages (List[str]): List of candidate passages for selection
            
        Returns:
            tuple: (selected_passages, total_tokens) - Final passages and token count
        """
        # Limit 1: Maximum number of chunks (priority)
        max_chunks = self.max_context_chunks
        selected_passages = passages[:max_chunks]
        
        # Limit 2: Token verification (safety)
        total_tokens = 0
        final_passages = []
        
        for i, passage in enumerate(selected_passages):
            token_estimate = self._estimate_tokens(passage)
            
            # Verify passage does not exceed limit
            if total_tokens + token_estimate > self.context_token_limit:
                print(f"Warning: Passage {i+1} ignored: token limit exceeded ({total_tokens + token_estimate} > {self.context_token_limit})")
                break
                
            final_passages.append(passage)
            total_tokens += token_estimate

        print(f"Dense S&C RAG - Selected passages: {len(final_passages)}/{len(passages)} (tokens: {total_tokens})")
        return final_passages, total_tokens

    def generate_answer(self, query: str, passages: List[str], 
                       reranked_metadata: Optional[List[Dict[str, Any]]] = None,
                       equipment_info: Optional[Dict] = None,
                       processed_query: Optional[Any] = None) -> str:
        """
        Generate response with multi-query dense symptom-cause knowledge graph integration.
        
        Performs comprehensive RAG generation with adaptive strategy selection based
        on context availability, relevance assessment, and multi-query processing
        capabilities specific to symptom-cause enriched structure.
        
        Args:
            query (str): User question (used if no processed_query)
            passages (List[str]): Selected passage texts
            reranked_metadata (Optional[List[Dict[str, Any]]]): Reranked document metadata
            equipment_info (Optional[Dict]): Equipment information for filtering
            processed_query (Optional[Any]): Complete LLM preprocessing data
            
        Returns:
            str: Generated response based on available context and strategy
        """
        # Strict selection with limits
        selected_passages, total_tokens = self.select_passages_with_limits(passages)
        context_rerank = "\n\n".join(selected_passages)

        # Document relevance evaluation
        if reranked_metadata:
            doc_relevance_stats = evaluate_document_relevance(
                reranked_metadata, 
                threshold=self.relevance_threshold
            )
            doc_has_content = doc_relevance_stats["is_relevant"]
        else:
            doc_has_content = len(selected_passages) > 0 and any(len(p.strip()) > 20 for p in selected_passages)
            doc_relevance_stats = {"is_relevant": doc_has_content, "max_score": 0.0}

        # Dense S&C KG context retrieval with multi-query or fallback
        try:
            # Multi-query if processed_query available
            if processed_query and hasattr(processed_query, 'query_variants'):
                print(f"Using Multi-Query Dense S&C KG with LLM preprocessing")
                context_graph = get_structured_context_dense_sc_with_multi_query(
                    processed_query.filtered_query,
                    processed_query.query_variants,
                    equipment_info or {},
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            elif equipment_info:
                print(f"Using Single-Query Dense S&C KG with equipment matching")
                context_graph = get_structured_context_dense_sc_with_equipment_filter(
                    query,
                    equipment_info,
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            else:
                print(f"Using Single-Query Dense S&C KG classic")
                context_graph = get_structured_context_dense_sc(
                    query, 
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            
            if not context_graph or context_graph.startswith("No relevant"):
                context_graph = "[No relevant information found in Dense S&C Knowledge Graph]"
                kg_has_content = False
                triplet_count = 0
            else:
                triplet_count = len([line for line in context_graph.split('\n') if '→' in line])
                print(f"Dense S&C KG context retrieved: {triplet_count} triplets")
                kg_has_content = triplet_count > 0
                
        except Exception as e:
            context_graph = f"[Unable to retrieve Dense S&C KG context: {str(e)}]"
            print(f"Error retrieving Dense S&C KG context: {e}")
            kg_has_content = False
            triplet_count = 0

        # Response strategy (similar to other generators)
        if not doc_has_content and not kg_has_content:
            print("Strategy: NO_CONTEXT")
            return "Information not available in the provided context."
        
        elif not doc_has_content and kg_has_content:
            prompt_type = "kg_only"
            mode_str = "Multi-Query" if processed_query and hasattr(processed_query, 'query_variants') else "Single-Query"
            print(f"Strategy: KG_DENSE_SC_ONLY ({mode_str}) - {triplet_count} enriched triplets")
        
        elif doc_has_content and not kg_has_content:
            prompt_type = "doc_only"
            print(f"Strategy: DOC_ONLY")
        
        else:
            prompt_type = "both"
            mode_str = "Multi-Query" if processed_query and hasattr(processed_query, 'query_variants') else "Single-Query"
            print(f"Strategy: HYBRID ({mode_str}) - docs + {triplet_count} Dense S&C triplets")

        # Prompt generation
        prompt = self._generate_adaptive_prompt(
            prompt_type, query, context_rerank, context_graph, doc_relevance_stats
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=self.max_tokens,
                stop=["\n\n", "\nQuestion"]
            )
            
            generated_answer = response.choices[0].message.content.strip()
            
            # Complete diagnostic logging with mode
            multi_query_info = ""
            if processed_query and hasattr(processed_query, 'query_variants'):
                multi_query_info = f" [Multi-Query: {len(processed_query.query_variants)} variants]"
            
            print(f"Dense S&C response generated: {len(generated_answer)} characters")
            print(f"Strategy used: {prompt_type.upper()}{multi_query_info}")
            print(f"Doc context: {'✅' if doc_has_content else '❌'} (max score: {doc_relevance_stats.get('max_score', 0):.3f})")
            print(f"Dense S&C KG context: {'✅' if kg_has_content else '❌'} ({triplet_count} triplets)")
            print(f"Passages used: {len(selected_passages)}, tokens: {total_tokens}")
            
            return generated_answer

        except Exception as e:
            error_msg = f"OpenAI API error with Dense S&C KG context: {str(e)}"
            print(error_msg)
            return error_msg

    def _generate_adaptive_prompt(self, prompt_type: str, query: str, 
                                 context_rerank: str, context_graph: str,
                                 doc_stats: Dict[str, Any]) -> str:
        """
        Generate adaptive prompt for dense symptom-cause knowledge graph context.
        
        Constructs context-appropriate prompts using externalized templates based
        on available context types and dense S&C structure requirements.
        
        Args:
            prompt_type (str): Type of prompt strategy to use
            query (str): User query
            context_rerank (str): Document context from reranking
            context_graph (str): Dense S&C knowledge graph context
            doc_stats (Dict[str, Any]): Document relevance statistics
            
        Returns:
            str: Complete formatted prompt for generation
            
        Raises:
            ValueError: If unknown prompt type is specified
        """
        
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(self.prompts.keys())}")
        
        template = self.prompts[prompt_type]
        
        if prompt_type == "kg_only":
            return template.format(
                context_graph=context_graph,
                query=query
            )
        elif prompt_type == "doc_only":
            return template.format(
                context_rerank=context_rerank,
                query=query
            )
        else:  # both
            return template.format(
                importance_context_rerank=self.importance_context_rerank,
                max_score=doc_stats.get('max_score', 0),
                context_rerank=context_rerank,
                importance_context_graph=self.importance_context_graph,
                max_triplets=self.max_triplets,
                context_graph=context_graph,
                query=query
            )

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive generator configuration statistics.
        
        Provides detailed configuration information for monitoring, debugging,
        and performance analysis specific to dense S&C KG integration.
        
        Returns:
            Dict[str, Any]: Complete generator configuration and capabilities
        """
        return {
            "model": self.model,
            "max_context_chunks": self.max_context_chunks,
            "context_token_limit": self.context_token_limit,
            "max_tokens": self.max_tokens,
            "max_triplets": self.max_triplets,
            "relevance_threshold": self.relevance_threshold,
            "kg_type": "dense_s&c",
            "structure": "symptom+cause enriched with semantic propagation",
            "prompts_loaded": len(self.prompts),
            "multi_query_support": True
        }