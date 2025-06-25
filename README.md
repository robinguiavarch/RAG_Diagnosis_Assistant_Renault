# Renault Diagnosis Assistant – Joint Project Télécom Paris & Renault

**Renault Supervisors:** Vincent Feuillard & François-Paul Servant
**Télécom Paris Supervisors:** Matthieu Labeau & Maria Boritchev

**Authors:** Nour Nounah, Nawfal Adil, Mohamed Limame Malainine, Habibata Samake, Robin Guiavarch

## Introduction

Renault Diagnosis Assistant (RDA) is an interactive Streamlit application designed to facilitate fast and efficient troubleshooting of equipment on Renault production lines. Intended for workers and technicians, the application simultaneously compares responses from four Retrieval-Augmented Generation (RAG) systems to provide optimal solutions in an industrial context.

In parallel, the Streamlit application also serves as a research and development tool for machine learning engineers. This tool allows an in-depth comparative analysis to identify the best design strategy for the final RAG, ensuring robust selection based on precise technical criteria.

Each RAG uses specific supplier documents related to equipment to generate relevant responses using a Large Language Model (LLM). Three out of four RAG systems enrich this documentary search approach using Knowledge Graphs, representing structured relationships between symptoms, causes, and remedies (SCR).

## Preprocessing Pipeline

The preprocessing pipeline consists of two main branches:

### 1. Documentary Branch

This branch focuses on directly utilizing supplier documents:

* **PDF extraction and cleaning** into JSON format.
* **Chunking** of extracted texts.
* **BM25 indexing** for lexical search.
* **Embedding and FAISS indexing** for semantic search.

### 2. Knowledge Graph Branch

This branch enhances the documentary approach through graph-based structuring:

* **Extraction of SCR triplets** (Symptom–Cause–Remedy) from documents into CSV format with metadata (page number, equipment).
* **Creation of three Knowledge Graphs (KG):**

  * **KG Sparse:** SCR nodes directly connected by their logical relations.
  * **KG Dense:** connects similar symptoms using a configurable hybrid distance (Jaccard + Cosine + Levenshtein).
  * **KG Dense S\&C:** connects symptom–cause combinations using the same hybrid approach.
* **BM25 and FAISS indexing** of symptoms for each KG.

## Search Workflows

### 1. Documentary Workflow

* **User query** processed by a query\_processor (ChatGPT) generating three condensed variants.
* **Matching:** retrieval of relevant chunks using lexical (BM25) and semantic (FAISS) indexes.
* **Fusion and re-ranking** of top results using MS-MARCO.
* **Context injection** (top 3 chunks) into an LLM generator, subject to a relevance threshold.

### 2. Knowledge Graph Workflow

* **User query** processed by a query\_processor (ChatGPT) producing condensed variants.
* **Matching:** symptom identification via lexical (BM25), semantic (FAISS), and Levenshtein distance.
* **Fusion of relevant SCR triplets**, enriched through Dense KG traversal when applicable.
* **Context injection** (top 3 SCR triplets) into an LLM generator, subject to a relevance threshold.

### 3. Combined Context

* The context transmitted to the LLM may be empty, contain only documentary chunks, only SCR triplets, or a combination thereof depending on their relevance.

## RAG Evaluation

An independent LLM judge evaluates responses provided by the four approaches on a scale from 0 to 5:

* **RAG 1:** Documentary approach only.
* **RAG 2:** Documentary + KG Sparse.
* **RAG 3:** Documentary + KG Dense.
* **RAG 4:** Documentary + KG Dense S\&C.

This evaluation ensures transparency regarding the comparative effectiveness of each method to best assist Renault technicians in their diagnostics. Specifically, our RAG comparator enables:

* **Ablation studies:** assessing the impact of removing the Knowledge Graph branch and graph densification.
* **Comparison of densification methods:** analyzing respective performances of the two Knowledge Graph densification methods (Dense vs. Dense S\&C) to determine the optimal strategy.

---

## I. Installation and Usage

### Prerequisites

You have two options for installing and running the application: **Poetry** or **Docker**.

**Important Note:** There are known compatibility issues between FAISS and macOS. Therefore, using Docker is strongly recommended for greater stability.

### Option 1: Installation with Poetry

**Install dependencies:**

```bash
poetry install
```

Ensure Poetry is installed. If not, consult the [official Poetry documentation](https://python-poetry.org/docs/) for installation guidance.

**Running scripts:**

To run a script using Poetry:

```bash
poetry run python script_name.py
```

### Option 2: Installation with Docker (Recommended)

#### Building the Docker Image:

Grant execution permissions to `docker-commands.sh` and build:

```bash
chmod +x docker-commands.sh
./docker-commands.sh build
```

#### Running the Streamlit Application:

To launch the application using Docker, use the simplified command provided in `docker-commands.sh`:

```bash
./docker-commands.sh streamlit
```

This command launches the application on the default port (`8502`).
For a custom port, use:

```bash
./docker-commands.sh streamlit-port 8503
```

Access the application via:
[http://localhost:8502](http://localhost:8502) (or specified custom port).

### Other Useful Docker Commands:

* **Clean Docker and stop running containers:**

```bash
./docker-commands.sh cleanup
```

* **Check occupied ports:**

```bash
./docker-commands.sh check-ports
```

For a complete list of available Docker commands, use:

```bash
./docker-commands.sh menu
```

---

### Running a Custom Python Script with Docker

If the specific command you want to use is not included in the `docker-commands.sh` script, you can directly execute a Python script via Docker using the following syntax:

```bash
docker run --rm \
    -v $(pwd)/.env:/app/.env \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/tests:/app/tests \
    --network host \
    diagnosis-app poetry run python path/script_name.py
```

---

## II. Project Architecture

```markdown
RAG_Diagnosis_Assistant_Renault/
│
├── streamlit_app.py                    # Main Streamlit application (final user interface)
│
├── config/                             # Application configuration and LLM prompts
│   ├── settings.yaml                   # Comprehensive configuration for KGs, LLM, and cloud services
│   └── prompts/                        # Externalized prompts for interactions with LLM
│
├── core/                               # Core business modules: query handling, evaluation, retrieval, and response generation
│   ├── cloud/                          # Cloud connection management
│   ├── query_processing/               # Pre-processing and handling of user queries
│   ├── evaluation/                     # Comparative evaluation by an independent LLM judge
│   ├── retrieval_engine/               # Lexical and semantic search engines
│   ├── retrieval_graph/                # Querying and retrieving information from Knowledge Graphs
│   ├── reranking_engine/               # Re-ranking of retrieved results
│   └── response_generation/            # Generation of final responses via LLM
│
├── data/                               # Comprehensive data structure
│   ├── documents/
│   │   ├── source_pdfs/                # Original PDF documents
│   │   ├── extracted_json/             # Cleaned and structured JSON files
│   │   └── processed_chunks/           # Optimized segments for search
│   │
│   ├── indexes/
│   │   ├── lexical_bm25/               # Whoosh index for keyword-based search
│   │   ├── semantic_faiss/             # FAISS index for vector-based search
│   │   └── embeddings/                 # Document and metadata embeddings
│   │
│   └── knowledge_base/
│       ├── scr_triplets/               # Extracted Symptom–Cause–Remedy triplets
│       ├── symptom_bm25_dense/         # BM25 symptom index for Dense KG
│       ├── symptom_bm25_sparse/        # BM25 symptom index for Sparse KG
│       ├── symptom_bm25_dense_sc/      # BM25 symptom index for Dense S&C KG
│       ├── symptom_embeddings_dense/   # Embeddings for standard Dense KG
│       ├── symptom_embeddings_sparse/  # Embeddings for Sparse KG
│       └── symptom_embeddings_dense_sc/ # Embeddings for Dense S&C KG
│
├── pipeline_step/                      # Pipeline scripts for preprocessing and Knowledge Graph construction
│   ├── data_extraction/                # Data extraction from PDFs
│   ├── data_processing/                # Preprocessing of extracted data
│   ├── index_building/                 # Construction of search indexes
│   └── knowledge_graph_setup/          # Creation and configuration of Knowledge Graphs
│
├── tests/                              # Unit and integration tests to ensure reliability
│
└── analytics/                          # Analytical tools and visualizations to evaluate system performance and quality
```

Each script and the `settings.yaml` file contain detailed explanations at the beginning for ease of understanding.

---

## III. Preprocessing Before Launching the Application

### Initial prerequisite

* Add your PDF document `doc-R-30iB.pdf` to the following folder:

```
data/documents/source_pdfs/
```

---

### A. Documentary Branch

Follow these steps to prepare documents for the documentary search:

1. **Extract PDF to JSON**

```bash
poetry run python pipeline_step/data_extraction/pdf_to_json.py
# Results stored in: data/documents/extracted_json/
```

2. **Adaptive chunk segmentation**

```bash
poetry run python pipeline_step/data_processing/adaptive_chunking.py
# Results stored in: data/documents/processed_chunks/
```

3. **Generate embeddings for the chunks**

```bash
poetry run python pipeline_step/data_processing/vector_embedding.py
# Results stored in: data/indexes/embeddings/
```

4. **Build the lexical index (BM25)**

```bash
poetry run python pipeline_step/index_building/build_lexical_index.py
# Results stored in: data/indexes/lexical_bm25/
```

5. **Build the semantic index (FAISS)**

```bash
poetry run python pipeline_step/index_building/build_semantic_index.py
# Results stored in: data/indexes/semantic_faiss/
```

---

### B. Knowledge Graph Branch

Follow these steps to build and prepare your Knowledge Graphs:

1. **Build the Knowledge Graphs**

* Dense KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_dense_knowledge_graph.py
```

* Sparse KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_sparse_knowledge_graph.py
```

* Dense S\&C KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_dense_s&c_knowledge_graph.py
```

2. **Create the BM25 symptom index**

* Dense KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense.py
```

* Sparse KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_sparse.py
```

* Dense S\&C KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense_s&c.py
```

3. **Create the FAISS symptom vector index**

* Dense KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense.py
```

* Sparse KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_sparse.py
```

* Dense S\&C KG:

```bash
poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense_s&c.py
```

---

### Additional Information

For now, the Knowledge Graphs are hosted on Neo4j Cloud but will be deleted on **7 July 2025**. Be sure to rebuild the Knowledge Graphs after that date if necessary.

---

## IV. RAG Workflows

The RAG workflows are organized around two main branches that converge towards response generation, with a final comparative evaluation system.

### General Architecture

The system implements **four distinct RAG approaches** that share a common foundation but differ in their use of Knowledge Graphs:

1. **Classic RAG**: Pure document branch
2. **RAG + Dense KG**: Document branch + KG with semantic propagation
3. **RAG + Sparse KG**: Document branch + KG with direct 1:1:1 structure
4. **RAG + Dense S&C KG**: Document branch + KG with Symptom+Cause enrichment

### 1. Document Branch (Common to all 4 RAG)

#### 1.1 Intelligent Preprocessing (`core/query_processing/`)

**`unified_query_processor.py`** - Unified entry point with LLM preprocessing
- LLM analysis of user query via **`llm_client.py`** (OpenAI GPT-4o)
- Technical term and error code extraction via **`response_parser.py`**
- Query variant generation for Multi-Query Fusion
- Equipment detection and matching via **`equipment_matcher.py`** (cosine similarity)

#### 1.2 Multi-Modal Retrieval (`core/retrieval_engine/`)

**`enhanced_retrieval_engine.py`** - Multi-variant search orchestrator
- Candidate collection across all query variants
- Intelligent deduplication by content hash
- Fusion and reranking with primary query

**`lexical_search.py`** - BM25 lexical search
- Whoosh index with optimized BM25F scoring
- Robust query cleaning to prevent parsing errors
- Enhanced metadata support (word_count, quality_score, source_file)

**`semantic_search.py`** - FAISS semantic search
- FAISS index with normalized SentenceTransformer embeddings
- Precise cosine similarity calculation (L2 distance → cosine correction)
- Support for legacy and new metadata formats

**`hybrid_fusion.py`** - Lexical and semantic result fusion
- Min-max normalization of BM25 and FAISS scores
- Configurable weighted fusion (α for lexical, 1-α for semantic)
- Content-based deduplication with provenance tracking

#### 1.3 Neural Reranking (`core/reranking_engine/`)

**`cross_encoder_reranker.py`** - CrossEncoder reranking
- ms-marco-MiniLM-L-6-v2 model with GPU/CPU optimization
- Batch processing for efficiency
- Preservation of fusion/BM25/FAISS scores for analysis

### 2. Knowledge Graph Branch (RAG 2, 3, 4)

#### 2.1 Multi-Query KG Retrieval (`core/retrieval_graph/`)

**`dense_kg_querier.py`** - Dense KG with semantic propagation
- Hybrid search BM25 + FAISS + Levenshtein via **`hybrid_symptom_matcher.py`**
- Multi-Query Fusion with MAX Score strategy across variants
- Equipment Matching for targeted triplet filtering
- Semantic propagation: one symptom → multiple causes/remedies

**`sparse_kg_querier.py`** - Sparse KG with 1:1:1 structure
- Pure FAISS search on symptoms with triplet_id metadata
- Guaranteed 1:1:1 structure: one symptom → one cause → one remedy
- No semantic propagation for perfect traceability

**`dense_sc_kg_querier.py`** - Dense S&C KG with combined enrichment
- Search on combined "symptom + cause" text for enriched matching
- Semantic propagation with symptom-cause contextual analysis
- combined_text metadata to improve semantic relevance

#### 2.2 Intelligent Equipment Matching

**`equipment_matcher.py`** - LLM ↔ KG matching
- Exact → partial → semantic matching (SentenceTransformer)
- Configurable similarity threshold (default: 0.9)
- Available equipment extraction via Neo4j queries

### 3. Response Generation (`core/response_generation/`)

#### 3.1 Specialized Generators

**`standard_rag_generator.py`** - Classic RAG
- OpenAI generation with document context only
- Strict chunk limit management (max_context_chunks)
- Externalized prompt templates

**`rag_with_kg_dense_generator.py`** - RAG + Dense KG
- Document relevance evaluation (CrossEncoder normalization)
- Adaptive strategies: DOC_ONLY, KG_ONLY, HYBRID, NO_CONTEXT
- Multi-Query support if processed_query available

**`rag_with_kg_sparse_generator.py`** - RAG + Sparse KG
- Similar logic to Dense but with 1:1:1 structure
- Preservation of direct symptom→cause→remedy traceability

**`rag_with_kg_dense_sc_generator.py`** - RAG + Dense S&C KG
- Enriched context with combined symptom-cause analysis
- Specialized prompts for S&C structure exploitation

#### 3.2 Generation Strategies

Each KG generator implements **adaptive strategy logic**:

1. **Relevance Assessment**: Sigmoid normalization of CrossEncoder scores
2. **Strategy Selection**:
  - `NO_CONTEXT`: Neither docs nor KG relevant
  - `DOC_ONLY`: Documents relevant, KG empty
  - `KG_ONLY`: KG relevant, documents below threshold
  - `HYBRID`: Both documents AND KG relevant
3. **Adaptive Prompts**: Externalized templates per strategy
4. **Multi-Query Management**: Use variants if processed_query available

### 4. Comparative Evaluation (`core/evaluation/`)

#### 4.1 LLM Judge System

**`response_evaluator.py`** - Comparative evaluation of 4 RAG approaches
- **`llm_judge_client.py`** - OpenAI client with consistency verification
- 0-5 scoring per approach with comparative justification
- Automatic retry on detected inconsistency
- Text similarity analysis for validation

### 5. Operation Modes

- **Classic Mode**: Single-query on all components
- **Multi-Query Mode**: LLM preprocessing + variants + MAX Score fusion
- **Equipment Matching**: Automatic KG filtering based on detected equipment
- **Cloud/Local**: Automatic Neo4j Cloud → Local fallback

This architecture enables **rigorous evaluation** of different RAG approaches with objective metrics, while providing the flexibility needed for optimization according to specific Renault industrial diagnosis requirements.

---

## V. Analytical Tools

The system provides a comprehensive suite of analytical tools for monitoring, evaluating, and visualizing the RAG diagnosis system components.

**`document_quality_analyzer.py`** - Evaluates processed JSON documents quality with SCR extraction validation, performs statistical content assessment, and provides diagnostic capabilities for identifying extraction method effectiveness and text integrity issues.

**`chunk_quality_analyzer.py`** - Assesses document segmentation quality by analyzing chunk structure, detecting concatenated words from poor extraction, and providing recommendations for improving text preprocessing workflows.

**`scr_extraction_analyzer.py`** - Provides statistical analysis of extracted SCR triplets from CSV files, including distribution metrics, equipment coverage analysis, and automated quality reporting with detailed extraction statistics.

**`dense_kg_visualizer.py`** - Creates network visualizations of Dense Knowledge Graphs with semantic propagation, featuring cloud/local Neo4j connectivity, equipment filtering, and comprehensive graph statistics including density metrics.

**`sparse_kg_visualizer.py`** - Visualizes Sparse Knowledge Graphs with emphasis on 1:1:1 linear structure validation, providing traceability analysis and structural integrity verification for direct symptom-cause-remedy mappings.

**`dense_sc_kg_visualizer.py`** - Specialized visualization tool for Dense Symptom & Cause Knowledge Graphs, analyzing combined symptom+cause text relationships and validating S&C enrichment characteristics with densification metrics.

---

## VI. Application Usage Guide

The Streamlit application is designed to compare four Retrieval-Augmented Generation (RAG) approaches: Classical, KG Dense, KG Sparse, and KG Dense S\&C. This guide will help you navigate the user interface and make the most of its available features.

### General Usage

1. **Enter your query:**

   * Input a question or issue related to your industrial equipment in the dedicated query field.

2. **Start the search:**

   * Click the search button to initiate the retrieval and comparison of answers from the four RAG systems.

### Displayed Results

The results are presented in four distinct columns, each corresponding to one of the RAG systems being compared:

* **Classical RAG**
* **RAG with KG Dense**
* **RAG with KG Sparse**
* **RAG with KG Dense S\&C**

Each column includes:

* **Generated Response:** The answer produced by the corresponding RAG system.
* **Execution Time Metrics:** Detailed timing of each generation step for each system.
* **"Show Details" Button:**

  * Reveals additional information such as:

    * The exact query sent to the LLM.
    * The specific context injected into the generative model.
    * The sources used (document chunks or SCR triplets) to generate the final response.

### Automatic Comparative Evaluation

An automatic evaluation is performed by a separate "LLM Judge," which:

* Assesses and rates each response on a scale from 0 to 5.
* Provides a quick and clear visualization of the comparative performance of each RAG approach.

### Additional Information

Many parameters can be configured in the `settings.yaml` file, including:

* The relevance threshold required for context injection into the generative LLM.
* The weights assigned to different hybrid similarity metrics.
* The number of retrieved chunks or triplets.

This flexibility allows for fine-tuning the retrieval and generation workflows according to specific evaluation or deployment needs.
