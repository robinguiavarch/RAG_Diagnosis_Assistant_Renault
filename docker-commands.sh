#!/bin/bash

"""
Docker Utility Scripts: RAG Project Container Management

This script provides comprehensive Docker utility functions for managing the RAG project
containerization. It implements Docker-based execution of various RAG system components
and is designed for development, testing, and deployment of the diagnosis application.

Key components:
- Image building: Automated Docker image construction and management
- Index creation: FAISS dense and sparse index generation via containers
- Testing utilities: Knowledge graph and retrieval system testing
- Streamlit deployment: Web application containerized deployment with flexible ports
- Neo4j integration: Cloud and local Neo4j database connectivity testing
- System diagnostics: Comprehensive health checks and port monitoring

Dependencies: Docker, Neo4j, Poetry, Streamlit
Usage: ./docker-commands.sh [command] [arguments]
"""

# Color configuration for display output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker configuration parameters
DOCKER_IMAGE="diagnosis-app"
DOCKER_BASE_CMD="docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data -v $(pwd)/config:/app/config -v $(pwd)/tests:/app/tests"

echo -e "${BLUE}🐳 UTILITAIRES DOCKER - PROJET RAG${NC}"
echo "=============================================="

build_image() {
    """
    Build the Docker image for the RAG application.
    
    Creates a Docker image with all necessary dependencies and configurations
    for running the RAG diagnosis application in a containerized environment.
    
    Returns:
        int: Exit code 0 for success, 1 for failure
        
    Raises:
        DockerError: If Docker build process fails
    """
    echo -e "${YELLOW}🔨 Construction de l'image Docker...${NC}"
    docker build -t $DOCKER_IMAGE .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Image Docker construite avec succès!${NC}"
    else
        echo -e "${RED}❌ Erreur lors de la construction de l'image${NC}"
        exit 1
    fi
}

create_symptom_faiss_dense_11() {
    """
    Create FAISS Dense index via Docker container execution.
    
    Executes the dense FAISS index creation script within a Docker container,
    providing isolated environment for vector index generation with proper
    network access for Neo4j connectivity.
    
    Returns:
        int: Exit code from the containerized index creation process
    """
    echo -e "${YELLOW}🚀 Création de l'index FAISS Dense via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_symptom_faiss_dense_11.py
}

create_symptom_faiss_sparse_12() {
    """
    Create FAISS Sparse index via Docker container execution.
    
    Executes the sparse FAISS index creation script within a Docker container,
    providing isolated environment for sparse vector index generation with
    network connectivity for database access.
    
    Returns:
        int: Exit code from the containerized sparse index creation process
    """
    echo -e "${YELLOW}🚀 Création de l'index FAISS Sparse via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_symptom_faiss_sparse_12.py
}

test_kg_dense_query() {
    """
    Execute knowledge graph dense query tests via Docker.
    
    Runs comprehensive testing of the dense knowledge graph query system
    within a containerized environment, passing through all command line
    arguments for flexible test configuration.
    
    Args:
        *args: Variable arguments passed to the test script
    
    Returns:
        int: Exit code from the test execution
    """
    echo -e "${YELLOW}🧪 Test kg_dense_query via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python tests/test_kg_dense_query.py "$@"
}

create_kg_dense_neo4j_09() {
    """
    Build the dense knowledge graph in Neo4j via Docker.
    
    Executes the knowledge graph construction script for dense graph creation
    within a Docker container with network access for Neo4j database operations.
    
    Returns:
        int: Exit code from the knowledge graph creation process
    """
    echo -e "${YELLOW}📚 Construction de la KG Dense (Neo4j) via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_kg_dense_neo4j_09.py
}

test_graph_retrieval() {
    """
    Execute graph retrieval testing via Docker container.
    
    Runs the graph retrieval testing script within a containerized environment
    with proper volume mounting for scripts directory and network access for
    database connectivity.
    
    Args:
        *args: Variable arguments passed to the graph retrieval test script
    
    Returns:
        int: Exit code from the graph retrieval test execution
    """
    echo -e "${YELLOW}🔍 Test Graph Retrieval via Docker...${NC}"
    $DOCKER_BASE_CMD --network host -v $(pwd)/scripts:/app/scripts $DOCKER_IMAGE poetry run python scripts/graph_retrieval_13.py "$@"
}

run_diagnostic() {
    """
    Execute comprehensive diagnostic tests via Docker.
    
    Runs a complete diagnostic suite for the RAG system within a Docker
    container, providing comprehensive health checks and system validation.
    Maintained for backward compatibility with existing workflows.
    
    Returns:
        int: Exit code from the diagnostic test execution
    """
    echo -e "${YELLOW}🔧 Diagnostic complet via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python tests/test_kg_dense_query.py --diagnostic
}

run_streamlit() {
    """
    Launch Streamlit application with Neo4j Cloud support and flexible port configuration.
    
    Deploys the Streamlit web application within a Docker container with proper
    volume mounting for application code and configuration. Supports flexible
    port configuration and Neo4j Cloud connectivity.
    
    Args:
        port (int, optional): Port number for Streamlit deployment (default: 8502)
    
    Returns:
        int: Exit code from the Streamlit application deployment
        
    Raises:
        DockerError: If container deployment fails
    """
    # Port determination with parameter or default
    local PORT=${2:-8502}  # Default port 8502 to avoid conflicts
    
    echo -e "${YELLOW}🌐 Lancement de Streamlit avec accès Neo4j Cloud...${NC}"
    echo -e "${BLUE}📱 L'application sera accessible sur http://localhost:${PORT}${NC}"
    echo -e "${GREEN}🌐 Mode Neo4j Cloud activé (NEO4J_CLOUD_ENABLED=true)${NC}"
    echo -e "${YELLOW}⚠️  Assurez-vous que votre .env contient les URLs Cloud Neo4j${NC}"
    
    # Corrected volume mounting for core directory and flexible port
    docker run --rm \
        -v $(pwd)/.env:/app/.env \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/config:/app/config \
        -v $(pwd)/core:/app/core \
        -p ${PORT}:8501 \
        --name streamlit-rag-app \
        $DOCKER_IMAGE
}

run_streamlit_port() {
    """
    Launch Streamlit application on custom port specified by user.
    
    Provides functionality to deploy Streamlit application on a user-specified
    port for avoiding conflicts or specific deployment requirements. Validates
    port parameter and provides clear usage instructions.
    
    Args:
        port (int): Custom port number for Streamlit deployment
    
    Returns:
        int: Exit code from the Streamlit deployment or 1 for invalid usage
        
    Raises:
        ParameterError: If port parameter is missing or invalid
    """
    if [ -z "$2" ]; then
        echo -e "${RED}❌ Usage: ./docker-commands.sh streamlit-port <PORT>${NC}"
        echo -e "${YELLOW}💡 Exemple: ./docker-commands.sh streamlit-port 8503${NC}"
        exit 1
    fi
    
    local PORT=$2
    echo -e "${YELLOW}🌐 Lancement de Streamlit sur port personnalisé ${PORT}...${NC}"
    echo -e "${BLUE}📱 L'application sera accessible sur http://localhost:${PORT}${NC}"
    
    docker run --rm \
        -v $(pwd)/.env:/app/.env \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/config:/app/config \
        -v $(pwd)/core:/app/core \
        -p ${PORT}:8501 \
        --name streamlit-rag-app-${PORT} \
        $DOCKER_IMAGE
}

run_custom() {
    """
    Execute custom command within Docker container environment.
    
    Provides flexibility to run arbitrary commands within the containerized
    RAG application environment with proper volume mounting and network access.
    Useful for debugging, maintenance, and custom operations.
    
    Args:
        *args: Variable arguments representing the custom command to execute
    
    Returns:
        int: Exit code from the custom command execution
    """
    echo -e "${YELLOW}⚡ Exécution de commande personnalisée via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE "$@"
}

cleanup_docker() {
    """
    Clean up Docker resources and stop running containers.
    
    Performs comprehensive cleanup of Docker resources including stopping
    active Streamlit containers, removing unused images, and system pruning
    for resource management and conflict resolution.
    
    Returns:
        int: Exit code 0 for successful cleanup
    """
    echo -e "${YELLOW}🧹 Nettoyage Docker...${NC}"
    
    # Stop running Streamlit containers
    docker stop streamlit-rag-app 2>/dev/null || true
    docker stop $(docker ps -q --filter "name=streamlit-rag-app") 2>/dev/null || true
    
    # General system cleanup
    docker system prune -f
    echo -e "${GREEN}✅ Nettoyage terminé${NC}"
}

check_ports() {
    """
    Check port availability for various application services.
    
    Performs comprehensive port availability checking for Streamlit application
    and Neo4j database services, helping to identify conflicts and plan
    deployment configurations.
    
    Returns:
        None: Prints port status information directly to stdout
    """
    echo -e "${YELLOW}🔍 Vérification des ports...${NC}"
    echo -e "${BLUE}Port 8501 (Streamlit défaut):${NC}"
    lsof -i :8501 2>/dev/null || echo "Port 8501 libre"
    echo -e "${BLUE}Port 8502 (Streamlit alternatif):${NC}"
    lsof -i :8502 2>/dev/null || echo "Port 8502 libre"
    echo -e "${BLUE}Port 8503 (Streamlit alternatif 2):${NC}"
    lsof -i :8503 2>/dev/null || echo "Port 8503 libre"
    echo ""
    echo -e "${BLUE}Ports Neo4j locaux (si utilisés):${NC}"
    echo -e "${BLUE}Port 7687 (Neo4j Bolt):${NC}"
    lsof -i :7687 2>/dev/null || echo "Port 7687 libre"
    echo -e "${BLUE}Port 7689 (Neo4j Sparse):${NC}"
    lsof -i :7689 2>/dev/null || echo "Port 7689 libre"
    echo -e "${BLUE}Port 7690 (Neo4j Dense S&C):${NC}"
    lsof -i :7690 2>/dev/null || echo "Port 7690 libre"
}

test_neo4j_cloud() {
    """
    Test Neo4j Cloud connectivity for all database instances.
    
    Executes comprehensive connectivity testing for all Neo4j Cloud instances
    (Dense, Sparse, Dense S&C) within a Docker container to validate database
    access and configuration before application deployment.
    
    Returns:
        int: Exit code from the connectivity test execution
        
    Raises:
        ConnectionError: If Neo4j Cloud connectivity fails
    """
    echo -e "${YELLOW}🌐 Test de connexion Neo4j Cloud...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python -c "
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Test Dense Cloud connectivity
print('🧠 Test Dense Cloud...')
try:
    uri = os.getenv('NEO4J_DENSE_CLOUD_URI')
    password = os.getenv('NEO4J_DENSE_CLOUD_PASS')
    if uri and password:
        driver = GraphDatabase.driver(uri, auth=('neo4j', password))
        with driver.session() as session:
            result = session.run('RETURN 1')
            print('✅ Dense Cloud: Connexion réussie')
        driver.close()
    else:
        print('❌ Dense Cloud: Variables manquantes')
except Exception as e:
    print(f'❌ Dense Cloud: {e}')

# Test Sparse Cloud connectivity
print('🟤 Test Sparse Cloud...')
try:
    uri = os.getenv('NEO4J_SPARSE_CLOUD_URI')
    password = os.getenv('NEO4J_SPARSE_CLOUD_PASS')
    if uri and password:
        driver = GraphDatabase.driver(uri, auth=('neo4j', password))
        with driver.session() as session:
            result = session.run('RETURN 1')
            print('✅ Sparse Cloud: Connexion réussie')
        driver.close()
    else:
        print('❌ Sparse Cloud: Variables manquantes')
except Exception as e:
    print(f'❌ Sparse Cloud: {e}')

# Test Dense S&C Cloud connectivity
print('🔶 Test Dense S&C Cloud...')
try:
    uri = os.getenv('NEO4J_DENSE_SC_CLOUD_URI')
    password = os.getenv('NEO4J_DENSE_SC_CLOUD_PASS')
    if uri and password:
        driver = GraphDatabase.driver(uri, auth=('neo4j', password))
        with driver.session() as session:
            result = session.run('RETURN 1')
            print('✅ Dense S&C Cloud: Connexion réussie')
        driver.close()
    else:
        print('❌ Dense S&C Cloud: Variables manquantes')
except Exception as e:
    print(f'❌ Dense S&C Cloud: {e}')
"
}

show_logs() {
    """
    Display Streamlit application logs from running container.
    
    Retrieves and displays logs from the active Streamlit container for
    debugging and monitoring purposes. Provides clear feedback when no
    container is currently running.
    
    Returns:
        None: Prints log information directly to stdout
    """
    echo -e "${YELLOW}📋 Affichage des logs Streamlit...${NC}"
    docker logs streamlit-rag-app 2>/dev/null || echo -e "${RED}❌ Aucun conteneur Streamlit en cours${NC}"
}

show_menu() {
    """
    Display the main menu with available commands and usage information.
    
    Provides comprehensive menu display with all available commands, usage
    examples, and helpful tips for Docker-based RAG system management.
    Includes color-coded sections for better readability.
    
    Returns:
        None: Prints menu information directly to stdout
    """
    echo ""
    echo -e "${BLUE}📋 COMMANDES DISPONIBLES:${NC}"
    echo "1.  build                          - Construire l'image Docker"
    echo "2.  create_symptom_faiss_dense_11  - Créer l'index FAISS Dense"
    echo "3.  create_symptom_faiss_sparse_12 - Créer l'index FAISS Sparse"
    echo "4.  test_kg_dense_query           - Tests kg_dense_query"
    echo "5.  test_graph_retrieval          - Test contexte LLM interactif"
    echo "6.  diagnostic                     - Diagnostic complet (raccourci)"
    echo "7.  streamlit                      - Lancer Streamlit sur port 8502 (Neo4j Cloud)"
    echo "8.  streamlit-port <PORT>          - Lancer Streamlit sur port personnalisé"
    echo "9.  custom <cmd>                   - Commande personnalisée"
    echo "10. cleanup                        - Nettoyer Docker + arrêter conteneurs"
    echo "11. create_kg_dense_neo4j_09       - Construire la KG dense sur Neo4j"
    echo "12. check-ports                    - Vérifier les ports occupés"
    echo "13. test-neo4j-cloud              - Tester connexion Neo4j Cloud"
    echo "14. logs                          - Afficher logs Streamlit"
    echo ""
    echo -e "${GREEN}🌐 MODE NEO4J CLOUD ACTIVÉ${NC}"
    echo -e "${YELLOW}💡 Conseils:${NC}"
    echo "   - Neo4j Cloud est activé par défaut (NEO4J_CLOUD_ENABLED=true)"
    echo "   - Utilisez 'test-neo4j-cloud' pour vérifier les connexions"
    echo "   - Port 8502 par défaut pour éviter les conflits"
    echo "   - Utilisez 'streamlit-port 8503' pour un port personnalisé"
    echo "   - Utilisez 'cleanup' en cas de problème avec les conteneurs"
    echo ""
}

# Main argument handling and command dispatch
case "$1" in
    "build")
        build_image
        ;;
    "create_symptom_faiss_dense_11")
        create_symptom_faiss_dense_11
        ;;
    "create_symptom_faiss_sparse_12")
        create_symptom_faiss_sparse_12
        ;;
    "create_kg_dense_neo4j_09")
        create_kg_dense_neo4j_09
        ;;
    "test_kg_dense_query")
        shift
        test_kg_dense_query "$@"
        ;;
    "test_graph_retrieval")
        shift
        test_graph_retrieval "$@"
        ;;
    "diagnostic")
        run_diagnostic
        ;;
    "streamlit")
        run_streamlit "$@"
        ;;
    "streamlit-port")
        run_streamlit_port "$@"
        ;;
    "custom")
        shift
        run_custom "$@"
        ;;
    "cleanup")
        cleanup_docker
        ;;
    "check-ports")
        check_ports
        ;;
    "test-neo4j-cloud")
        test_neo4j_cloud
        ;;
    "logs")
        show_logs
        ;;
    "menu"|"")
        show_menu
        ;;
    *)
        echo -e "${RED}❌ Commande inconnue: $1${NC}"
        show_menu
        exit 1
        ;;
esac