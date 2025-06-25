"""
RAG Comparator Application: Multi-System RAG Evaluation Platform

This module provides a comprehensive Streamlit web application for comparing four different
RAG (Retrieval-Augmented Generation) systems. It implements a complete comparison interface
with multi-query fusion capabilities and is designed for research and development evaluation
of various RAG approaches including classic RAG and Knowledge Graph enhanced variants.

Key components:
- Multi-system comparison: Classic RAG, Dense KG, Sparse KG, Dense S&C KG
- Query processing: LLM-based query optimization and multi-query fusion
- Equipment matching: Automatic equipment detection and filtering
- Response evaluation: LLM judge for comparative analysis
- Hybrid retrieval: BM25 + FAISS with cross-encoder reranking
- Performance metrics: Execution time tracking and analysis

Dependencies: Streamlit, OpenAI, Neo4j, FAISS, sentence-transformers, PyYAML
Usage: streamlit run streamlit_app.py
"""

# Standard imports
import yaml
import os
import time
import math
from dotenv import load_dotenv

# Streamlit configuration must be first
import streamlit as st
st.set_page_config(page_title="RAG Comparator 4 Columns", layout="wide")

# Project imports after Streamlit configuration
from core.retrieval_engine.lexical_search import BM25Retriever
from core.retrieval_engine.semantic_search import FAISSRetriever
from core.retrieval_engine.hybrid_fusion import fuse_results
from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
from core.response_generation.standard_rag_generator import OpenAIGenerator
from core.response_generation.rag_with_kg_dense_generator import OpenAIGeneratorKG
from core.response_generation.rag_with_kg_sparse_generator import OpenAIGeneratorKGSparse
from core.response_generation.rag_with_kg_dense_sc_generator import OpenAIGeneratorKGDenseSC

# LLM filter and judge imports
from core.query_processing import (
    create_query_processor, 
    create_enhanced_retrieval_engine
)
from core.evaluation import create_response_evaluator

# Knowledge Graph multi-query and equipment matching imports
from core.retrieval_graph.dense_kg_querier import (
    get_structured_context, 
    get_structured_context_with_equipment_filter
)
from core.retrieval_graph.sparse_kg_querier import (
    get_structured_context_sparse, 
    get_structured_context_sparse_with_equipment_filter
)
from core.retrieval_graph.dense_sc_kg_querier import (
    get_structured_context_dense_sc, 
    get_structured_context_dense_sc_with_equipment_filter
)

# Application title and description
st.title("🧠 Outil R&D RAG Comparator")
st.markdown("**🆕 Version Multi-Query** - Comparez RAG Classique, KG Dense, KG Sparse et KG Dense S&C avec LLM preprocessing intelligent")

# Cloud status display
cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
if cloud_enabled:
    st.success("🌐 Mode Cloud Neo4j activé")
else:
    st.info("🏠 Mode Local Neo4j")

# Load environment and configuration
load_dotenv()

@st.cache_data
def load_settings():
    """
    Load application settings from YAML configuration file.
    
    Reads the main configuration file containing paths, model names, and system
    parameters required for all RAG components initialization.
    
    Returns:
        dict: Parsed configuration dictionary with all application settings
        
    Raises:
        FileNotFoundError: If the settings.yaml file does not exist
        yaml.YAMLError: If the YAML file contains syntax errors
    """
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

settings = load_settings()
paths = settings["paths"]
models = settings["models"]
rerank_cfg = settings["reranking"]
gen_cfg = settings["generation"]

@st.cache_resource
def load_retrievers():
    """
    Load and initialize BM25 and FAISS retrievers with proper configuration paths.
    
    Initializes both lexical (BM25) and semantic (FAISS) retrieval systems using
    the paths specified in the configuration file. Includes fallback logic for
    metadata file location and comprehensive error handling.
    
    Returns:
        tuple: (BM25Retriever, FAISSRetriever) initialized retriever instances
        
    Raises:
        FileNotFoundError: If required index files are not found
        Exception: Various initialization errors with detailed messages
    """
    try:
        # Initialize BM25 retriever with configured index directory
        bm25 = BM25Retriever(index_dir=paths["bm25_index"])
        
        # Initialize FAISS retriever with standard paths
        faiss_index_path = paths["faiss_index"]  # data/indexes/semantic_faiss/index.faiss
        faiss_metadata_path = paths["embedding_file"]  # data/indexes/embeddings/metadata.pkl
        
        # Fallback logic for metadata file location
        if not os.path.exists(faiss_metadata_path):
            # Try alternative metadata location in FAISS directory
            alternative_metadata = os.path.join(paths["faiss_index_dir"], "metadata.pkl")
            if os.path.exists(alternative_metadata):
                faiss_metadata_path = alternative_metadata
                print(f"Using alternative metadata path: {alternative_metadata}")
            else:
                raise FileNotFoundError(f"FAISS metadata not found. Check:\n"
                                      f"  - {faiss_metadata_path}\n"
                                      f"  - {alternative_metadata}\n"
                                      f"Run FAISS index creation script first.")
        
        faiss = FAISSRetriever(
            index_path=faiss_index_path, 
            metadata_path=faiss_metadata_path
        )
        
        print(f"Retrievers loaded successfully:")
        print(f"   BM25: {paths['bm25_index']}")
        print(f"   FAISS: {faiss_index_path}")
        print(f"   Metadata: {faiss_metadata_path}")
        
        return bm25, faiss
        
    except Exception as e:
        st.error(f"❌ Erreur chargement retrievers: {e}")
        st.info("💡 Assurez-vous d'avoir créé les index BM25 et FAISS avec les scripts appropriés")
        raise

@st.cache_resource
def load_reranker():
    """
    Load and initialize the CrossEncoder reranker with optimized parameters.
    
    Creates a local CrossEncoder instance for document reranking using the
    configured model with optimized parameters for performance and accuracy.
    
    Returns:
        CrossEncoderReranker: Initialized reranker instance with configured model
        
    Raises:
        Exception: Model loading or initialization errors
    """
    return CrossEncoderReranker(
        model_name=models.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        max_length=1024
    )

@st.cache_resource
def load_generators():
    """
    Load and initialize all four RAG response generators.
    
    Creates instances of all four RAG generation systems: classic RAG,
    dense KG, sparse KG, and dense S&C KG generators with consistent
    configuration parameters.
    
    Returns:
        tuple: Four generator instances (classic, dense_kg, sparse_kg, dense_sc_kg)
        
    Raises:
        Exception: Generator initialization or configuration errors
    """
    # Classic RAG generator
    classic_generator = OpenAIGenerator(
        model=models.get("openai_model", "gpt-4o"),
        context_token_limit=6000,
        max_tokens=gen_cfg.get("max_new_tokens", 512),
    )
    
    # Dense Knowledge Graph generator
    kg_dense_generator = OpenAIGeneratorKG(
        model=models.get("openai_model", "gpt-4o"),
        context_token_limit=6000
    )
    
    # Sparse Knowledge Graph generator
    kg_sparse_generator = OpenAIGeneratorKGSparse(
        model=models.get("openai_model", "gpt-4o"),
        context_token_limit=6000
    )
    
    # Dense S&C Knowledge Graph generator
    kg_dense_sc_generator = OpenAIGeneratorKGDenseSC(
        model=models.get("openai_model", "gpt-4o"),
        context_token_limit=6000
    )
    
    return classic_generator, kg_dense_generator, kg_sparse_generator, kg_dense_sc_generator

@st.cache_resource
def load_llm_preprocessing():
    """
    Load and initialize the LLM preprocessing system for query optimization.
    
    Creates query processor and enhanced retrieval engine for multi-query fusion
    and intelligent query preprocessing capabilities.
    
    Returns:
        tuple: (query_processor, enhanced_retrieval) or (None, None) if failed
        
    Raises:
        Exception: LLM preprocessing system initialization errors
    """
    try:
        query_processor = create_query_processor()
        
        bm25, faiss = load_retrievers()
        reranker = load_reranker()
        
        enhanced_retrieval = create_enhanced_retrieval_engine(
            bm25_retriever=bm25,
            faiss_retriever=faiss,
            reranker=reranker
        )
        
        return query_processor, enhanced_retrieval
        
    except Exception as e:
        st.error(f"❌ Erreur chargement LLM preprocessing: {e}")
        return None, None

@st.cache_resource
def load_response_evaluator():
    """
    Load and initialize the LLM-based response evaluation system.
    
    Creates an evaluator instance for comparative analysis of the four RAG
    system responses using LLM-based judging capabilities.
    
    Returns:
        ResponseEvaluator or None: Initialized evaluator instance or None if failed
        
    Raises:
        Exception: Evaluator initialization errors
    """
    try:
        return create_response_evaluator()
    except Exception as e:
        st.error(f"❌ Erreur chargement évaluateur: {e}")
        return None

# Load all system components
bm25, faiss = load_retrievers()
reranker = load_reranker()
classic_generator, kg_dense_generator, kg_sparse_generator, kg_dense_sc_generator = load_generators()

# Load LLM preprocessing system
query_processor, enhanced_retrieval = load_llm_preprocessing()

# Load response evaluation system
response_evaluator = load_response_evaluator()

def rerank_with_cross_encoder(query, docs, top_k=3):
    """
    Perform local reranking using CrossEncoder to replace HuggingFace API.
    
    Applies cross-encoder reranking to the provided documents using the local
    CrossEncoder model, providing more accurate relevance scoring than initial
    retrieval methods.
    
    Args:
        query (str): User query for relevance calculation
        docs (list): List of document dictionaries to rerank
        top_k (int): Number of top documents to return after reranking
    
    Returns:
        list: Reranked documents with cross-encoder scores and metadata
        
    Raises:
        Exception: Reranking process errors with fallback to fused scores
    """
    try:
        reranked_docs = reranker.rerank(
            query=query,
            candidates=docs,
            top_k=top_k,
            return_scores=True
        )
        
        formatted_results = []
        for doc in reranked_docs:
            formatted_results.append({
                "text": doc["text"],
                "score": doc["cross_encoder_score"],
                "cross_encoder_score": doc["cross_encoder_score"],
                "fused_score": doc.get("fused_score", 0.0),
                "source": doc.get("source", "Unknown"),
                "original_rank": doc.get("original_rank", 0)
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error during CrossEncoder reranking: {e}")
        fallback_docs = sorted(docs, key=lambda x: x.get("fused_score", 0), reverse=True)[:top_k]
        return [{"text": d["text"], "score": d.get("fused_score", 0.0), "source": d.get("source", "Unknown")} for d in fallback_docs]

def determine_strategy_info(reranked_docs, kg_triplets_text, seuil_pertinence):
    """
    Determine and format strategy information for display purposes.
    
    Analyzes the relevance of documents and knowledge graph content to determine
    the optimal strategy (document-only, KG-only, hybrid, or no context) and
    formats the information for user interface display.
    
    Args:
        reranked_docs (list): List of reranked document results
        kg_triplets_text (str): Knowledge graph triplets text content
        seuil_pertinence (float): Relevance threshold for strategy determination
    
    Returns:
        dict: Strategy information with text, color, and details for display
    """
    
    kg_has_content = "Triplet" in kg_triplets_text if kg_triplets_text else False
    triplet_count = len([line for line in kg_triplets_text.split('\n') if 'Triplet' in line]) if kg_triplets_text else 0
    
    if reranked_docs:
        max_raw_score = max([d.get("cross_encoder_score", 0) for d in reranked_docs])
        try:
            max_normalized_score = 1.0 / (1.0 + math.exp(-max_raw_score))
        except:
            max_normalized_score = 0.5
        
        doc_has_content = max_normalized_score >= seuil_pertinence
    else:
        max_normalized_score = 0.0
        doc_has_content = False
    
    if not doc_has_content and not kg_has_content:
        return {
            "strategy": "AUCUN_CONTEXTE",
            "text": "🚫 **Stratégie :** AUCUN_CONTEXTE",
            "color": "red",
            "details": "Aucun contexte pertinent trouvé"
        }
    elif not doc_has_content and kg_has_content:
        return {
            "strategy": "KG_SEULEMENT", 
            "text": f"🧠 **Stratégie :** KG_SEULEMENT ",
            "color": "blue",
            "details": f"Knowledge Graph uniquement "
        }
    elif doc_has_content and not kg_has_content:
        return {
            "strategy": "DOC_SEULEMENT",
            "text": f"📄 Stratégie : DOC_SEULEMENT ",
            "color": "green", 
            "details": "Documents de recherche uniquement"
        }
    else:
        return {
            "strategy": "HYBRIDE",
            "text": f"🔄 Stratégie : HYBRIDE ",
            "color": "purple",
            "details": f"Documents + Knowledge Graph"
        }

def run_rag_system(query, system_type, use_llm_preprocessing):
    """
    Execute a RAG system with multi-query fusion capabilities if LLM preprocessing is enabled.
    
    Runs one of the four RAG systems (classic, dense, sparse, dense_sc) with optional
    LLM preprocessing for query optimization, multi-query fusion, and equipment matching.
    Handles both classic and enhanced retrieval pipelines.
    
    Args:
        query (str): User query to process
        system_type (str): Type of system ("classic", "dense", "sparse", "dense_sc")
        use_llm_preprocessing (bool): Whether to use LLM preprocessing and multi-query
    
    Returns:
        dict: Complete system execution results including answer, timing, and metadata
        
    Raises:
        Exception: Various system execution errors with fallback handling
    """
    
    # Variables for storing LLM data
    equipment_info = None
    processed_query = None

    # Unified LLM preprocessing for all systems
    if use_llm_preprocessing and query_processor and enhanced_retrieval:
        # LLM preprocessing + multi-query pipeline
        try:
            print(f"Activating Multi-Query for system: {system_type}")
            
            # Store processed_query
            processed_query = query_processor.process_user_query(query)
            retrieval_result = enhanced_retrieval.search_with_variants(processed_query)
            reranked = retrieval_result.chunks
            processing_time = retrieval_result.processing_time
            
            # Store complete metadata
            reranked_metadata = [{"content": d["text"], "score": d.get("cross_encoder_score", d.get("fused_score", 0.0)), "source": d["source"]} for d in reranked]

            
            # Extract equipment_info for KG systems
            equipment_info = {
                'primary_equipment': processed_query.equipment_info.primary_equipment,
                'equipment_type': processed_query.equipment_info.equipment_type,
                'manufacturer': processed_query.equipment_info.manufacturer,
                'series': processed_query.equipment_info.series
            }
            
            print(f"Multi-Query activated: {len(processed_query.query_variants)} variants")
            print(f"Equipment detected: {equipment_info['primary_equipment']}")
            
            # Retrieve KG triplets for all systems (even if empty for classic)
            kg_triplets_detailed = "\n".join([
                f"Triplet {i}: {t.get('symptom', '')} → {t.get('cause', '')} → {t.get('remedy', '')}"
                for i, t in enumerate(retrieval_result.triplets, 1)
            ]) if retrieval_result.triplets else ""
            
        except Exception as e:
            st.error(f"❌ Erreur LLM preprocessing pour {system_type}: {e}")
            # Fallback to classic pipeline
            use_llm_preprocessing = False
            equipment_info = None
            processed_query = None
    
    if not use_llm_preprocessing or not query_processor:
        # Classic unified pipeline for all 4 systems
        print(f"Classic mode for system: {system_type}")
        start_time = time.time()
        
        # BM25 and FAISS search (identical for all systems)
        bm25_raw = bm25.search(query, top_k=3)
        for doc in bm25_raw:
            doc["source"] = "Lexical (BM25)"

        faiss_raw = faiss.search(query, top_k=3)
        for doc in faiss_raw:
            doc["source"] = "Sémantique (FAISS)"

        # Fusion and local reranking (identical for all systems)
        fused = fuse_results(bm25_raw, faiss_raw, top_k=rerank_cfg.get("top_k_before_rerank", 10))
        reranked = rerank_with_cross_encoder(query, fused, top_k=rerank_cfg.get("final_top_k", 3))
        
        processing_time = time.time() - start_time
        reranked_metadata = [{"content": d["text"], "score": d.get("cross_encoder_score", d.get("fused_score", 0.0)), "source": d["source"]} for d in reranked]

        kg_triplets_detailed = ""

   # Define effective query once
    effective_query = processed_query.filtered_query if (use_llm_preprocessing and processed_query) else query

    # Store context for debug display
    document_context = "\n\n".join([d["text"] for d in reranked[:gen_cfg.get("max_context_chunks", 3)]])
    kg_context = ""

    # Generation according to system type with filtered query
    try:
        if system_type == "classic":
            # Classic RAG: only documents, no KG
            answer = classic_generator.generate_answer(effective_query, [d["text"] for d in reranked])
            kg_context = "[Pas de contexte KG pour RAG Classique]"
            
        elif system_type == "dense":
            answer = kg_dense_generator.generate_answer(
                effective_query, [d["text"] for d in reranked], 
                reranked_metadata=reranked, 
                equipment_info=equipment_info,
                processed_query=processed_query if use_llm_preprocessing else None
            )
            # Retrieve KG context for display
            if processed_query and hasattr(processed_query, 'query_variants'):
                from core.retrieval_graph.dense_kg_querier import get_structured_context_with_multi_query
                kg_context = get_structured_context_with_multi_query(
                    processed_query.filtered_query, processed_query.query_variants, 
                    equipment_info or {}, format_type="detailed", max_triplets=3
                )
            elif equipment_info:
                kg_context = get_structured_context_with_equipment_filter(
                    effective_query, equipment_info, format_type="detailed", max_triplets=3
                )
            else:
                kg_context = get_structured_context(effective_query, format_type="detailed", max_triplets=3)
            
        elif system_type == "sparse":
            answer = kg_sparse_generator.generate_answer(
                effective_query, [d["text"] for d in reranked], 
                reranked_metadata=reranked,
                equipment_info=equipment_info,
                processed_query=processed_query if use_llm_preprocessing else None
            )
            # Retrieve KG context for display
            if processed_query and hasattr(processed_query, 'query_variants'):
                from core.retrieval_graph.sparse_kg_querier import get_structured_context_sparse_with_multi_query
                kg_context = get_structured_context_sparse_with_multi_query(
                    processed_query.filtered_query, processed_query.query_variants, 
                    equipment_info or {}, format_type="detailed", max_triplets=3
                )
            elif equipment_info:
                kg_context = get_structured_context_sparse_with_equipment_filter(
                    effective_query, equipment_info, format_type="detailed", max_triplets=3
                )
            else:
                kg_context = get_structured_context_sparse(effective_query, format_type="detailed", max_triplets=3)
            
        elif system_type == "dense_sc":
            answer = kg_dense_sc_generator.generate_answer(
                effective_query, [d["text"] for d in reranked], 
                reranked_metadata=reranked,
                equipment_info=equipment_info,
                processed_query=processed_query if use_llm_preprocessing else None
            )
            # Retrieve KG context for display
            if processed_query and hasattr(processed_query, 'query_variants'):
                from core.retrieval_graph.dense_sc_kg_querier import get_structured_context_dense_sc_with_multi_query
                kg_context = get_structured_context_dense_sc_with_multi_query(
                    processed_query.filtered_query, processed_query.query_variants, 
                    equipment_info or {}, format_type="detailed", max_triplets=3
                )
            elif equipment_info:
                kg_context = get_structured_context_dense_sc_with_equipment_filter(
                    effective_query, equipment_info, format_type="detailed", max_triplets=3
                )
            else:
                kg_context = get_structured_context_dense_sc(effective_query, format_type="detailed", max_triplets=3)
            
        else:
            answer = f"❌ Système inconnu: {system_type}"

    except Exception as e:
        answer = f"❌ Erreur génération {system_type}: {str(e)}"
        kg_context = f"❌ Erreur récupération contexte KG: {str(e)}"

    # Retrieve KG triplets after generation for display if not already done
    if system_type in ["dense", "sparse", "dense_sc"] and not kg_triplets_detailed:
        try:
            # Retrieve for display with equipment matching if available
            if system_type == "dense":
                if equipment_info:
                    kg_triplets_detailed = get_structured_context_with_equipment_filter(
                        query, equipment_info, format_type="detailed", max_triplets=3
                    )
                else:
                    kg_triplets_detailed = get_structured_context(query, format_type="detailed", max_triplets=3)
                    
            elif system_type == "sparse":
                if equipment_info:
                    kg_triplets_detailed = get_structured_context_sparse_with_equipment_filter(
                        query, equipment_info, format_type="detailed", max_triplets=3
                    )
                else:
                    kg_triplets_detailed = get_structured_context_sparse(query, format_type="detailed", max_triplets=3)
                    
            elif system_type == "dense_sc":
                if equipment_info:
                    kg_triplets_detailed = get_structured_context_dense_sc_with_equipment_filter(
                        query, equipment_info, format_type="detailed", max_triplets=3
                    )
                else:
                    kg_triplets_detailed = get_structured_context_dense_sc(query, format_type="detailed", max_triplets=3)
        except Exception:
            kg_triplets_detailed = "Erreur récupération triplets KG"

    # Strategy for KG systems
    if system_type in ["dense", "sparse", "dense_sc"] and kg_triplets_detailed:
        strategy_info = determine_strategy_info(reranked_metadata, kg_triplets_detailed, gen_cfg.get("seuil_pertinence", 0.7))
    else:
        strategy_info = {"strategy": "RAG_CLASSIQUE", "text": "📄 RAG Classique", "color": "blue"}

    # Add multi-query info to strategy
    if processed_query and use_llm_preprocessing:
        strategy_info["text"] += f" [Multi-Query: {len(processed_query.query_variants)} variantes]"

    return {
        "answer": answer,
        "time": processing_time,
        "docs_count": len(reranked),
        "reranked_docs": reranked_metadata,
        "kg_triplets": kg_triplets_detailed,
        "strategy": strategy_info,
        "equipment_info": equipment_info,
        "processed_query": processed_query,
        "effective_query": effective_query,
        "document_context": document_context,
        "kg_context": kg_context
    }

# User interface
query = st.text_area("💬 Votre requête", height=100, 
                    placeholder="Ex: motor overheating FANUC R-30iB error ACAL-006")

# Display options
col1, col2 = st.columns(2)
with col1:
    show_details = st.checkbox("Afficher les détails techniques", value=False)
with col2:
    use_llm_preprocessing = st.checkbox("🧠 Activer Multi-Query Fusion + Equipment Matching", value=True, 
                                       help="Utilise un LLM pour optimiser la requête + Multi-Query sur les KG + détection d'équipement")

if st.button("🚀 Comparer les 4 systèmes", type="primary") and query.strip():
    
    # Execute all 4 systems with multi-query fusion
    with st.spinner("⚡ Génération des 4 réponses avec Multi-Query Fusion..."):
        # Sequential execution to maintain sophisticated logic
        classic_result = run_rag_system(query, "classic", use_llm_preprocessing)
        dense_result = run_rag_system(query, "dense", use_llm_preprocessing)
        sparse_result = run_rag_system(query, "sparse", use_llm_preprocessing)
        dense_sc_result = run_rag_system(query, "dense_sc", use_llm_preprocessing)

    # Display multi-query info if LLM preprocessing active
    if use_llm_preprocessing and dense_result.get("processed_query"):
        processed_query = dense_result["processed_query"]
        st.markdown("---")
        st.markdown("### 🧠 Multi-Query Fusion (LLM Preprocessing)")
        
        col_mq1, col_mq2, col_mq3 = st.columns(3)
        with col_mq1:
            st.info(f"**Query filtrée:** {processed_query.filtered_query}")
        with col_mq2:
            st.info(f"**Variantes:** {len(processed_query.query_variants)}")
        with col_mq3:
            if processed_query.query_variants:
                st.info(f"**Première variante:** {processed_query.query_variants[0][:50]}...")
        
        # Display complete variants
        if show_details:
            with st.expander("🔍 Détails Multi-Query"):
                st.write("**Query originale:**", query)
                st.write("**Query filtrée:**", processed_query.filtered_query)
                st.write("**Variantes générées:**")
                for i, variant in enumerate(processed_query.query_variants, 1):
                    st.write(f"  {i}. {variant}")

    # Display equipment info if LLM preprocessing active
    if use_llm_preprocessing and dense_result.get("equipment_info"):
        equipment_info = dense_result["equipment_info"]
        st.markdown("### 🏭 Equipment Detection (Multi-Query)")
        col_eq1, col_eq2, col_eq3, col_eq4 = st.columns(4)
        with col_eq1:
            st.info(f"**Equipment:** {equipment_info.get('primary_equipment', 'Unknown')}")
        with col_eq2:
            st.info(f"**Type:** {equipment_info.get('equipment_type', 'Unknown')}")
        with col_eq3:
            st.info(f"**Manufacturer:** {equipment_info.get('manufacturer', 'Unknown')}")
        with col_eq4:
            st.info(f"**Series:** {equipment_info.get('series', 'Unknown')}")

    # Four-column display
    st.markdown("---")
    mode_title = "Multi-Query Fusion" if use_llm_preprocessing else "Mode Classique"
    st.markdown(f"## 📊 Comparaison des 4 systèmes - {mode_title}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📘 RAG Classique")
        st.markdown("*BM25 + FAISS + CrossEncoder*")
        st.success(classic_result["answer"])
        st.caption(f"⏱️ {classic_result['time']:.2f}s | 📄 {classic_result['docs_count']} docs")
        
        # Strategy display
        strategy = classic_result["strategy"]
        st.markdown(f"<p style='color: {strategy['color']}; font-size: 0.8em;'>{strategy['text']}</p>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🧠 RAG + KG Dense")
        mode_str = "Multi-Query + Equipment" if use_llm_preprocessing else "Single-Query + Equipment"
        st.markdown(f"*{mode_str}*")
        st.success(dense_result["answer"])
        st.caption(f"⏱️ {dense_result['time']:.2f}s | 📄 {dense_result['docs_count']} docs")
        
        # Strategy display
        strategy = dense_result["strategy"]
        st.markdown(f"<p style='color: {strategy['color']}; font-size: 0.8em;'>{strategy['text']}</p>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🟤 RAG + KG Sparse")
        mode_str = "Multi-Query + Equipment" if use_llm_preprocessing else "Single-Query + Equipment"
        st.markdown(f"*{mode_str}*")
        st.success(sparse_result["answer"])
        st.caption(f"⏱️ {sparse_result['time']:.2f}s | 📄 {sparse_result['docs_count']} docs")
        
        # Strategy display
        strategy = sparse_result["strategy"]
        st.markdown(f"<p style='color: {strategy['color']}; font-size: 0.8em;'>{strategy['text']}</p>", 
                   unsafe_allow_html=True)
    
    with col4:
        st.markdown("### 🔶 RAG + KG Dense S&C")
        mode_str = "Multi-Query + Equipment" if use_llm_preprocessing else "Single-Query + Equipment"
        st.markdown(f"*{mode_str}*")
        st.success(dense_sc_result["answer"])
        st.caption(f"⏱️ {dense_sc_result['time']:.2f}s | 📄 {dense_sc_result['docs_count']} docs")
        
        # Strategy display
        strategy = dense_sc_result["strategy"]
        st.markdown(f"<p style='color: {strategy['color']}; font-size: 0.8em;'>{strategy['text']}</p>", 
                   unsafe_allow_html=True)

    # LLM judge evaluation for 4 responses
    if response_evaluator:
        st.markdown("---")
        st.markdown("## 🤖 Analyse LLM Juge - 4 Systèmes")
        
        with st.spinner("🔍 Évaluation comparative des 4 réponses..."):
            try:
                eval_result = response_evaluator.evaluate_4_responses(
                    query,
                    classic_result["answer"],
                    dense_result["answer"],
                    sparse_result["answer"],
                    dense_sc_result["answer"]
                )
                
                # Score display
                col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
                
                with col_eval1:
                    score = eval_result.get("score_classic", 0)
                    st.metric("📘 Score Classique", f"{score:.1f}/5")
                
                with col_eval2:
                    score = eval_result.get("score_dense", 0)
                    st.metric("🧠 Score Dense", f"{score:.1f}/5")
                
                with col_eval3:
                    score = eval_result.get("score_sparse", 0)
                    st.metric("🟤 Score Sparse", f"{score:.1f}/5")
                
                with col_eval4:
                    score = eval_result.get("score_dense_sc", 0)
                    st.metric("🔶 Score Dense S&C", f"{score:.1f}/5")
                
                # Comparative analysis
                if "comparative_analysis" in eval_result:
                    st.info(f"**Analyse :** {eval_result['comparative_analysis']}")
                
                if "best_approach" in eval_result:
                    st.success(f"**Recommandation :** {eval_result['best_approach']}")
                    
            except Exception as e:
                st.warning(f"⚠️ Évaluation indisponible: {e}")

    # Execution time metrics
    st.markdown("---")
    st.markdown("## ⏱️ Temps d'exécution")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📘 RAG Classique", f"{classic_result['time']:.2f}s")
    
    with col2:
        st.metric("🧠 RAG KG Dense", f"{dense_result['time']:.2f}s")
    
    with col3:
        st.metric("🟤 RAG KG Sparse", f"{sparse_result['time']:.2f}s")
    
    with col4:
        st.metric("🔶 RAG KG Dense S&C", f"{dense_sc_result['time']:.2f}s")

    # Additional line for mode and cloud status
    col5, col6 = st.columns(2)
    with col5:
        st.metric("🌐 Mode", "Cloud" if cloud_enabled else "Local")
    with col6:
        multi_query_status = "✅ Multi-Query" if use_llm_preprocessing else "❌ Classique"
        st.metric("🧠 Mode", multi_query_status)

    # Conditional technical details display
    if show_details:
        st.markdown("---")
        st.markdown("## 🔍 Détails techniques - Contexte LLM complet")
        
        # Tabs for the 4 systems
        tab1, tab2, tab3, tab4 = st.tabs(["📘 Classique", "🧠 Dense", "🟤 Sparse", "🔶 Dense S&C"])
        
        for tab, result, system_name in zip([tab1, tab2, tab3, tab4], 
                                           [classic_result, dense_result, sparse_result, dense_sc_result],
                                           ["Classique", "Dense", "Sparse", "Dense S&C"]):
            with tab:
                # Display query used by generator for debugging
                st.markdown(f"### 🎯 Query utilisée par le générateur {system_name}")
                effective_query = result.get("effective_query", "N/A")
                if effective_query == query:
                    st.info(f"**Query originale utilisée:** {effective_query}")
                else:
                    st.success(f"**Query filtrée utilisée:** {effective_query}")
                    st.caption(f"*Query originale:* {query}")
                
                # Multi-query info if applicable
                if result.get("processed_query") and use_llm_preprocessing and system_name != "Classique":
                    st.markdown(f"### 🧠 Multi-Query Info - {system_name}")
                    pq = result["processed_query"]
                    st.info(f"Query filtrée: {pq.filtered_query}")
                    st.info(f"Variantes: {', '.join(pq.query_variants[:2])}...")
                
                # Display document context sent to LLM
                st.markdown(f"### 📄 Contexte documentaire envoyé au LLM {system_name}")
                document_context = result.get("document_context", "")
                if document_context and document_context.strip():
                    with st.expander(f"Contexte documentaire ({len(document_context)} caractères)"):
                        st.text_area("Contexte documentaire", value=document_context, height=200, 
                                        key=f"doc_context_{system_name}", label_visibility="hidden")
                else:
                    st.warning("Aucun contexte documentaire")
                
                # Display KG context sent to LLM if applicable
                if system_name != "Classique":
                    st.markdown(f"### 🧠 Contexte Knowledge Graph envoyé au LLM {system_name}")
                    kg_context = result.get("kg_context", "")
                    if kg_context and kg_context.strip() and not kg_context.startswith("[Pas de contexte"):
                        mode_info = " (Multi-Query)" if use_llm_preprocessing else " (Single-Query)"
                        with st.expander(f"Contexte KG {system_name}{mode_info} ({len(kg_context)} caractères)"):
                            st.text_area("Contexte Knowledge Graph", value=kg_context, height=300, 
                                        key=f"kg_context_{system_name}", label_visibility="hidden")
                    else:
                        st.warning(f"Aucun contexte KG pertinent pour {system_name}")
                
                # Detailed retrieved documents
                st.markdown(f"### 📊 Documents récupérés - {system_name}")
                for i, doc in enumerate(result["reranked_docs"]):
                    score = doc.get('score', 'N/A')
                    with st.expander(f"Document #{i+1} - {doc.get('source', 'Unknown')} (Score: {score:.3f})"):
                        st.markdown(doc['content'])
                
                # Detailed KG triplets
                if result["kg_triplets"] and system_name != "Classique":
                    st.markdown(f"### 🔗 Triplets KG détaillés - {system_name}")
                    mode_info = " (Multi-Query)" if use_llm_preprocessing else " (Single-Query)"
                    with st.expander(f"Triplets extraits du KG{mode_info}"):
                        st.text(result["kg_triplets"])

# Sidebar with information
with st.sidebar:
    st.markdown("## ℹ️ Comparateur RAG 4 Systèmes")
    
    if use_llm_preprocessing:
        st.markdown("### 🆕 Multi-Query Fusion ACTIVÉ")
        st.markdown("""
        **🧠 LLM Preprocessing + Multi-Query :**
        - Analyse LLM intelligente de la requête
        - Extraction de termes techniques
        - 🏭 Identification + Matching équipement
        - 🔄 Génération de variantes de requête
        - 📊 Recherche multi-variantes sur KG
        - ✂️ Déduplication intelligente
        - 🎯 Fusion avec stratégie MAX Score
        - 🔧 Filtrage KG par équipement détecté
        """)
    else:
        st.markdown("### 📄 Mode Classique ACTIVÉ")
        st.markdown("""
        **📄 Recherche classique :**
        - Recherche directe sur la requête
        - BM25 + FAISS + CrossEncoder
        - Equipment Matching disponible
        - Single-Query sur les KG
        """)
    
    st.markdown("""
    **4 Systèmes Comparés :**
    
    📘 **RAG Classique**
    - BM25 (recherche lexicale)
    - FAISS (recherche sémantique)
    - Fusion + CrossEncoder
    - Génération LLM
    
    🧠 **RAG + KG Dense**
    - Recherche vectorielle + hybride
    - Propagation sémantique
    - Traversée SCR enrichie
    - Multi-Query + Equipment Matching
    
    🟤 **RAG + KG Sparse**
    - Structure 1:1:1 directe
    - Traçabilité parfaite
    - Pas de propagation
    - Multi-Query + Equipment Matching
    
    🔶 **RAG + KG Dense S&C**
    - Symptôme + Cause combiné
    - Enrichissement contextuel
    - Propagation sémantique S&C
    - Multi-Query + Equipment Matching
    """)
    
    st.markdown("## 🤖 LLM Juge 4 Réponses")
    st.markdown("""
    **Évaluation automatique :**
    - Note individuelle (0-5) par système
    - Précision technique
    - Complétude de la réponse
    - Clarté pour ouvriers
    - Considérations sécurité
    
    **Comparaison intelligente** des 4 approches RAG.
    """)
    
    if use_llm_preprocessing:
        st.markdown("## 🧠 Multi-Query Fusion")
        st.markdown("""
        **Fonctionnement :**
        - LLM analyse et optimise la requête
        - Génération de 2 variantes intelligentes
        - Recherche parallèle sur KG avec toutes les variantes
        - Fusion avec stratégie MAX Score
        - Déduplication automatique
        - Sélection des top 3 triplets optimaux
        
        **Avantage :** Couverture sémantique élargie
        """)
    
    st.markdown("## 🏭 Equipment Matching")
    st.markdown("""
    **Fonctionnement :**
    - LLM extrait equipment de la requête
    - Cosine similarity avec KG equipment
    - Si match > 0.9 : filtrage ciblé KG
    - Sinon : recherche globale KG
    
    **Optimisation :** Recherche plus précise selon l'équipement identifié.
    """)
    
    if cloud_enabled:
        st.markdown("## 🌐 Status Cloud")
        st.success("Neo4j Cloud Actif")
        st.markdown("*Fallback automatique vers local si erreur*")
    else:
        st.markdown("## 🏠 Status Local")
        st.info("Neo4j Local Actif")
    
    if use_llm_preprocessing and query_processor:
        st.markdown("## 🧠 LLM Configuration")
        try:
            llm_config = query_processor.get_config() if query_processor else {}
            st.markdown(f"""
            **Configuration active :**
            - Modèle : `{llm_config.get('llm_config', {}).get('model', 'N/A')}`
            - Tokens max : `{llm_config.get('llm_config', {}).get('max_tokens', 'N/A')}`
            - Température : `{llm_config.get('llm_config', {}).get('temperature', 'N/A')}`
            - Multi-Query : ✅ ACTIVÉ
            """)
        except:
            st.markdown("*Configuration non disponible*")
    
    # Multi-query status
    st.markdown("---")
    if use_llm_preprocessing:
        st.success("🚀 Multi-Query Fusion ACTIVÉ")
        st.markdown("Les 3 KG utilisent la recherche Multi-Query")
    else:
        st.info("📄 Mode Single-Query")
        st.markdown("Recherche classique sur les KG")
    
    # Debug info
    if show_details:
        st.markdown("---")
        st.markdown("## 🔍 Mode Debug Actif")
        st.markdown("""
        **Détails techniques affichés :**
        - 🎯 Query utilisée par chaque générateur
        - 📄 Contexte documentaire envoyé au LLM
        - 🧠 Contexte KG envoyé au LLM
        - 📊 Documents récupérés détaillés
        - 🔗 Triplets KG extraits
        """)
        st.info("Ces informations vous permettent de vérifier que la query filtrée est bien utilisée et de voir exactement le contexte envoyé aux LLM générateurs.")