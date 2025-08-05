#!/bin/bash

# LUKi Memory Service Development Server
# Starts the FastAPI development server with hot reload

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting LUKi Memory Service Development Server${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment found. Creating one...${NC}"
    python -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Activated virtual environment (venv)${NC}"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Activated virtual environment (.venv)${NC}"
fi

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    pip install -e ".[dev]"
    touch .deps_installed
    echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# Check for required environment variables
if [ -z "$EMBEDDING_MODEL_NAME" ]; then
    export EMBEDDING_MODEL_NAME="all-MiniLM-L12-v2"
    echo -e "${YELLOW}⚠️  Using default embedding model: $EMBEDDING_MODEL_NAME${NC}"
fi

if [ -z "$CHROMA_PERSIST_DIR" ]; then
    export CHROMA_PERSIST_DIR="./data/chroma_db"
    echo -e "${YELLOW}⚠️  Using default ChromaDB directory: $CHROMA_PERSIST_DIR${NC}"
fi

# Create data directories
mkdir -p data/chroma_db
mkdir -p data/logs
mkdir -p models/sentence_transformers

echo -e "${GREEN}✅ Data directories ready${NC}"

# Set development environment variables
export API_DEBUG=true
export LOG_LEVEL=DEBUG
export CORS_ORIGINS="http://localhost:3000,http://localhost:8000,http://localhost:8080"

# Start the development server
echo -e "${BLUE}🌐 Starting server on http://localhost:8001${NC}"
echo -e "${YELLOW}📝 API docs will be available at http://localhost:8001/docs${NC}"
echo -e "${YELLOW}🔧 Press Ctrl+C to stop the server${NC}"

# Use uvicorn with reload for development
uvicorn luki_memory.api.http:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --reload-dir luki_memory \
    --log-level debug \
    --access-log
