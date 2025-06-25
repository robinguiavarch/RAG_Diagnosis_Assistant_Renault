#!/usr/bin/env python3
"""
test_sparse_kg_querier
======================

Diagnostic script for validating the connectivity and functionality of the
**Sparse** Neo4j knowledge-graph instance used by the project.

The script performs three independent checks:

1. **Environment variables** – verifies that ``NEO4J_URI_SPARSE``,
   ``NEO4J_USER_SPARSE`` and ``NEO4J_PASS_SPARSE`` are defined.
2. **Connection test** – attempts to establish a Neo4j session with the
   credentials above or those stored in *config/settings.yaml* (all candidate
   configurations are tried sequentially).
3. **Functionality test** – imports the public API of
   :pymod:`core.retrieval_graph.sparse_kg_querier` and runs a few basic
   queries to ensure that expected data can be retrieved.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase

# --------------------------------------------------------------------------- #
# Project import path & environment                                           #
# --------------------------------------------------------------------------- #

# Make 'src' importable when the script is executed from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from a local .env file if present
load_dotenv()

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def load_settings() -> Dict[str, Any]:
    """Return the content of *config/settings.yaml* as a Python dictionary."""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_neo4j_sparse_connection() -> Optional[Dict[str, str]]:
    """Attempt to connect to the Sparse Neo4j instance with every configuration.

    The function first prints the values of ``NEO4J_*`` environment variables
    for convenience, then iterates over the following credential sources:

    1. Environment variables.
    2. *config/settings.yaml* (if the ``neo4j_sparse`` section exists).

    The *first* set of credentials that succeeds is returned; otherwise
    ``None`` is returned.

    Returns
    -------
    dict | None
        The working configuration dictionary on success, *None* otherwise.
    """

    print("NEO4J SPARSE CONNECTION DIAGNOSTIC")
    print("=" * 50)

    # Display current environment variable values
    print("Sparse environment variables:")
    print(f"   NEO4J_URI_SPARSE  : {os.getenv('NEO4J_URI_SPARSE', 'NOT DEFINED')}")
    print(f"   NEO4J_USER_SPARSE : {os.getenv('NEO4J_USER_SPARSE', 'NOT DEFINED')}")
    print(f"   NEO4J_PASS_SPARSE : {'******' if os.getenv('NEO4J_PASS_SPARSE') else 'NOT DEFINED'}")
    print()

    # Collect candidate configurations
    candidates: list[dict[str, str]] = []

    env_candidate = {
        "uri": os.getenv("NEO4J_URI_SPARSE"),
        "user": os.getenv("NEO4J_USER_SPARSE"),
        "password": os.getenv("NEO4J_PASS_SPARSE"),
    }
    if all(env_candidate.values()):
        candidates.append(env_candidate)

    settings = load_settings()
    if settings and "neo4j_sparse" in settings:
        candidates.append(settings["neo4j_sparse"])

    # Try every candidate in order
    for idx, config in enumerate(candidates, start=1):
        print(f"Attempt {idx}: uri={config['uri']}, user={config['user']}")
        try:
            driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"]),
            )
            with driver.session() as session:
                greeting = session.run("RETURN 'ok' AS status").single()["status"]
                print(f"   Connection status: {greeting}")

                # Minimal data integrity check
                result = session.run(
                    """
                    MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE s.triplet_id = c.triplet_id AND c.triplet_id = r.triplet_id
                    RETURN count(*) AS valid_triplets
                    """
                )
                print(f"   1:1:1 triplets present: {result.single()['valid_triplets']}")
            driver.close()
            return config
        except Exception as exc:  # broad exception is intentional for diagnostics
            print(f"   Connection failed: {exc}\n")

    print("No working Sparse configuration found.")
    return None


def test_sparse_querier_functionality() -> bool:
    """Import and exercise the public API of *sparse_kg_querier*.

    The function runs three elementary queries that should always succeed when
    the graph has been built correctly. A failure usually indicates that the
    graph is missing or incomplete rather than an issue with the Python code.

    Returns
    -------
    bool
        *True* if the three test queries succeed, *False* otherwise.
    """

    print("\nSPARSE KG QUERIER FUNCTIONALITY TEST")
    print("=" * 50)

    try:
        # Lazy import – the module itself connects to Neo4j
        from core.retrieval_graph.sparse_kg_querier import (
            get_structured_context_sparse,
            get_similar_symptoms_sparse,
            get_possible_causes_sparse,
        )
    except ImportError as exc:
        print(f"Module import error: {exc}")
        print("Ensure that *core/retrieval_graph/sparse_kg_querier.py* exists and is importable.")
        return False

    # --------------------------------------------------------------------- #
    # Test 1: structured context                                            #
    # --------------------------------------------------------------------- #
    context = get_structured_context_sparse("broken belt")
    if not context:
        print("Test 1 failed – no structured context returned.")
        return False
    print("Test 1 passed.")

    # --------------------------------------------------------------------- #
    # Test 2: symptom similarity                                            #
    # --------------------------------------------------------------------- #
    similar = get_similar_symptoms_sparse("broken belt")
    if not similar:
        print("Test 2 failed – no similar symptoms returned.")
        return False
    print("Test 2 passed.")

    # --------------------------------------------------------------------- #
    # Test 3: possible causes                                               #
    # --------------------------------------------------------------------- #
    causes = get_possible_causes_sparse("broken belt")
    if not causes:
        print("Test 3 failed – no causes returned.")
        return False
    print("Test 3 passed.")

    print("All functionality tests passed.")
    return True


def get_neo4j_sparse_info() -> None:
    """Print hints to help the user initialise the Sparse knowledge graph."""
    print("\nNEO4J SPARSE INITIALISATION GUIDE")
    print("=" * 50)

    print("Things to verify:")
    print("  1. The Sparse database is created and running (port 7689).")

    print("\nRequired build scripts:")
    print("  • pipeline_step/knowledge_graph_setup/build_sparse_knowledge_graph.py")
    print("  • pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_sparse.py")
    print("  • pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_sparse.py")

    print("\nRequired artefacts:")
    print("  • data/knowledge_base/symptom_embeddings_sparse/index.faiss")
    print("  • data/knowledge_base/symptom_embeddings_sparse/symptom_embedding_sparse.pkl")

    print("\nSparse KG characteristics:")
    print("  • 1:1:1 structure (no semantic propagation).")
    print("  • Duplicate preservation through *triplet_id*.")
    print("  • Direct CSV → Neo4j import.")


def main() -> None:
    """CLI entry point."""
    working_cfg = test_neo4j_sparse_connection()
    if working_cfg is None:
        get_neo4j_sparse_info()
        sys.exit(1)

    print("\nWorking configuration:")
    print(f"   NEO4J_URI_SPARSE  = {working_cfg['uri']}")
    print(f"   NEO4J_USER_SPARSE = {working_cfg['user']}")
    print(f"   NEO4J_PASS_SPARSE = {'*' * len(working_cfg['password'])}\n")

    if test_sparse_querier_functionality():
        print("\nSparse KG querier is fully operational.")
    else:
        print("\nSparse KG querier is operational but some tests failed.")


if __name__ == "__main__":
    main()
