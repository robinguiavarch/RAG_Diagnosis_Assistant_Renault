#!/usr/bin/env python3
"""
test_dense_kg_querier
=====================

Diagnostic script for validating the connectivity and functionality of the
**Dense** Neo4j knowledge-graph instance used by the project.

The script performs two independent checks:

1. **Environment variables** – verifies that ``NEO4J_URI_DENSE``,
   ``NEO4J_USER_DENSE`` and ``NEO4J_PASS_DENSE`` are defined.
2. **Connection test** – attempts to establish a Neo4j session with the
   credentials above or those stored in *config/settings.yaml* (all candidate
   configurations are tried sequentially).
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


def test_neo4j_connection() -> Optional[Dict[str, str]]:
    """Attempt to connect to the Dense Neo4j instance with every configuration.

    The function first prints the values of ``NEO4J_*`` environment variables
    for convenience, then iterates over the following credential sources:

    1. Environment variables.
    2. *config/settings.yaml* (if the ``neo4j_dense`` section exists).

    The *first* set of credentials that succeeds is returned; otherwise
    ``None`` is returned.

    Returns
    -------
    dict | None
        The working configuration dictionary on success, *None* otherwise.
    """

    print("NEO4J DENSE CONNECTION DIAGNOSTIC")
    print("=" * 50)

    # Display current environment variable values
    print("Dense environment variables:")
    print(f"   NEO4J_URI_DENSE  : {os.getenv('NEO4J_URI_DENSE', 'NOT DEFINED')}")
    print(f"   NEO4J_USER_DENSE : {os.getenv('NEO4J_USER_DENSE', 'NOT DEFINED')}")
    print(f"   NEO4J_PASS_DENSE : {'******' if os.getenv('NEO4J_PASS_DENSE') else 'NOT DEFINED'}")
    print()

    # Collect candidate configurations
    candidates: list[dict[str, str]] = []

    env_candidate = {
        "uri": os.getenv("NEO4J_URI_DENSE"),
        "user": os.getenv("NEO4J_USER_DENSE"),
        "password": os.getenv("NEO4J_PASS_DENSE"),
    }
    if all(env_candidate.values()):
        candidates.append(env_candidate)

    settings = load_settings()
    if settings and "neo4j_dense" in settings:
        candidates.append(settings["neo4j_dense"])

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
            driver.close()
            return config
        except Exception as exc:  # broad exception is intentional for diagnostics
            print(f"   Connection failed: {exc}\n")

    print("No working Dense configuration found.")
    return None


def get_neo4j_dense_info() -> None:
    """Print hints to help the user initialise the Dense knowledge graph."""
    print("\nNEO4J DENSE INITIALISATION GUIDE")
    print("=" * 50)

    print("Things to verify:")
    print("  1. The Dense database is created and running (port 7688).")

    print("\nRequired build scripts:")
    print("  • pipeline_step/knowledge_graph_setup/build_dense_knowledge_graph.py")
    print("  • pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense.py")
    print("  • pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense.py")

    print("\nRequired artefacts:")
    print("  • data/knowledge_base/symptom_embeddings_dense/index.faiss")
    print("  • data/knowledge_base/symptom_embeddings_dense/symptom_embedding_dense.pkl")

    print("\nDense KG characteristics:")
    print("  • Semantic propagation across multiple nodes.")
    print("  • Synonym and concept merging activated.")
    print("  • Requires preprocessing before import.")


def main() -> None:
    """CLI entry point."""
    working_cfg = test_neo4j_connection()
    if working_cfg is None:
        get_neo4j_dense_info()
        sys.exit(1)

    print("\nWorking configuration:")
    print(f"   NEO4J_URI_DENSE  = {working_cfg['uri']}")
    print(f"   NEO4J_USER_DENSE = {working_cfg['user']}")
    print(f"   NEO4J_PASS_DENSE = {'*' * len(working_cfg['password'])}\n")


if __name__ == "__main__":
    main()
