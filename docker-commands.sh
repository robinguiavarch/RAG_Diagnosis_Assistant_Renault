#!/bin/bash
# Scripts utilitaires pour Docker - Projet RAG
# 🆕 VERSION CORRIGÉE avec support Cloud Neo4j et port flexible

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration Docker
DOCKER_IMAGE="diagnosis-app"
DOCKER_BASE_CMD="docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data -v $(pwd)/config:/app/config -v $(pwd)/tests:/app/tests"

echo -e "${BLUE}🐳 UTILITAIRES DOCKER - PROJET RAG${NC}"
echo "=============================================="

# Fonction pour construire l'image Docker
build_image() {
    echo -e "${YELLOW}🔨 Construction de l'image Docker...${NC}"
    docker build -t $DOCKER_IMAGE .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Image Docker construite avec succès!${NC}"
    else
        echo -e "${RED}❌ Erreur lors de la construction de l'image${NC}"
        exit 1
    fi
}

# Fonction pour créer l'index FAISS Dense
create_symptom_faiss_dense_11() {
    echo -e "${YELLOW}🚀 Création de l'index FAISS Dense via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_symptom_faiss_dense_11.py
}

# Fonction pour créer l'index FAISS Sparse (futur)
create_symptom_faiss_sparse_12() {
    echo -e "${YELLOW}🚀 Création de l'index FAISS Sparse via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_symptom_faiss_sparse_12.py
}

# Fonction pour lancer les tests
test_kg_dense_query() {
    echo -e "${YELLOW}🧪 Test kg_dense_query via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python tests/test_kg_dense_query.py "$@"
}

# Fonction pour créer la Knowledge Graph Dense (Neo4j)
create_kg_dense_neo4j_09() {
    echo -e "${YELLOW}📚 Construction de la KG Dense (Neo4j) via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python scripts/create_kg_dense_neo4j_09.py
}

# Fonction pour tester le graph retrieval
test_graph_retrieval() {
    echo -e "${YELLOW}🔍 Test Graph Retrieval via Docker...${NC}"
    $DOCKER_BASE_CMD --network host -v $(pwd)/scripts:/app/scripts $DOCKER_IMAGE poetry run python scripts/graph_retrieval_13.py "$@"
}

# Fonction pour le diagnostic (conservé pour compatibilité)
run_diagnostic() {
    echo -e "${YELLOW}🔧 Diagnostic complet via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python tests/test_kg_dense_query.py --diagnostic
}

# 🆕 FONCTION STREAMLIT CORRIGÉE avec support port flexible et Neo4j Cloud
run_streamlit() {
    # Détermination du port (paramètre ou défaut)
    local PORT=${2:-8502}  # Port 8502 par défaut pour éviter conflits
    
    echo -e "${YELLOW}🌐 Lancement de Streamlit avec accès Neo4j Cloud...${NC}"
    echo -e "${BLUE}📱 L'application sera accessible sur http://localhost:${PORT}${NC}"
    echo -e "${GREEN}🌐 Mode Neo4j Cloud activé (NEO4J_CLOUD_ENABLED=true)${NC}"
    echo -e "${YELLOW}⚠️  Assurez-vous que votre .env contient les URLs Cloud Neo4j${NC}"
    
    # 🔧 CORRECTION: Volume core au lieu de src + port flexible
    docker run --rm \
        -v $(pwd)/.env:/app/.env \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/config:/app/config \
        -v $(pwd)/core:/app/core \
        -p ${PORT}:8501 \
        --name streamlit-rag-app \
        $DOCKER_IMAGE
}

# 🆕 FONCTION STREAMLIT AVEC PORT PERSONNALISÉ
run_streamlit_port() {
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

# Fonction pour exécuter une commande personnalisée
run_custom() {
    echo -e "${YELLOW}⚡ Exécution de commande personnalisée via Docker...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE "$@"
}

# Fonction pour nettoyer Docker
cleanup_docker() {
    echo -e "${YELLOW}🧹 Nettoyage Docker...${NC}"
    
    # Arrêter les conteneurs Streamlit en cours
    docker stop streamlit-rag-app 2>/dev/null || true
    docker stop $(docker ps -q --filter "name=streamlit-rag-app") 2>/dev/null || true
    
    # Nettoyage général
    docker system prune -f
    echo -e "${GREEN}✅ Nettoyage terminé${NC}"
}

# 🆕 FONCTION POUR VÉRIFIER LES PORTS OCCUPÉS
check_ports() {
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

# 🆕 FONCTION POUR TESTER LA CONNEXION NEO4J CLOUD
test_neo4j_cloud() {
    echo -e "${YELLOW}🌐 Test de connexion Neo4j Cloud...${NC}"
    $DOCKER_BASE_CMD --network host $DOCKER_IMAGE poetry run python -c "
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Test Dense Cloud
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

# Test Sparse Cloud
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

# Test Dense S&C Cloud
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

# 🆕 FONCTION POUR AFFICHER LES LOGS STREAMLIT
show_logs() {
    echo -e "${YELLOW}📋 Affichage des logs Streamlit...${NC}"
    docker logs streamlit-rag-app 2>/dev/null || echo -e "${RED}❌ Aucun conteneur Streamlit en cours${NC}"
}

# Menu principal
show_menu() {
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

# Gestion des arguments
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