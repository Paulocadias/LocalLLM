#!/bin/bash
# LocalLLM Quick Start Script
# This script helps you verify installation and get started quickly

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════╗"
echo "║   LocalLLM Quick Start Verification           ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
echo "[1/9] Checking environment configuration..."
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}📝 Please edit .env and add your GitHub token:${NC}"
    echo "   GITHUB_TOKEN=your_token_here"
    echo ""
    echo "Get your token from: https://github.com/settings/tokens"
    echo ""
else
    echo -e "${GREEN}✓ .env file found${NC}"
fi

# Check Docker
echo "[2/9] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker Desktop.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check Ollama
echo "[3/9] Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${RED}✗ Ollama not accessible at localhost:11434${NC}"
    echo "  Please ensure Ollama is installed and running."
    exit 1
fi

# Check for required models
echo "[4/9] Checking Ollama models..."
REQUIRED_MODEL="qwen2.5-coder:7b"
if ollama list 2>/dev/null | grep -q "$REQUIRED_MODEL"; then
    echo -e "${GREEN}✓ Model $REQUIRED_MODEL found${NC}"
else
    echo -e "${YELLOW}⚠ Model $REQUIRED_MODEL not found${NC}"
    echo "  Pull it with: ollama pull $REQUIRED_MODEL"
fi

# Check if services are running
echo "[5/9] Checking Docker services..."
if docker ps --format "{{.Names}}" | grep -q "localllm-enhanced"; then
    echo -e "${GREEN}✓ Services are running${NC}"
else
    echo -e "${YELLOW}⚠ Services not running. Starting...${NC}"
    docker-compose -f docker-compose-simplified.yml up -d
    echo "Waiting 30 seconds for services to start..."
    sleep 30
fi

# Check service health
echo "[6/9] Checking service health..."

check_service() {
    local name=$1
    local url=$2
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ $name${NC}"
        return 0
    else
        echo -e "  ${YELLOW}⚠ $name (may need more time)${NC}"
        return 1
    fi
}

check_service "LocalLLM" "http://localhost:8080/health"
check_service "Vector DB" "http://localhost:8007/health"
check_service "Meta-Orchestrator" "http://localhost:8004/health"
check_service "Web Search" "http://localhost:8006/health"
check_service "Model Manager" "http://localhost:8005/health"

# Get API key
echo "[7/9] Getting API key..."
API_KEY=$(curl -s http://localhost:8080/admin/current-key 2>/dev/null | grep -o '"current_api_key":"[^"]*"' | cut -d'"' -f4)
if [ -n "$API_KEY" ]; then
    echo -e "${GREEN}✓ API Key: $API_KEY${NC}"
    echo "  Save this key for API requests!"
else
    echo -e "${YELLOW}⚠ Could not retrieve API key${NC}"
fi

# Test basic chat
echo "[8/9] Testing basic chat..."
if [ -n "$API_KEY" ]; then
    RESPONSE=$(curl -s -X POST http://localhost:8080/chat \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"message": "Say hello!", "model": "qwen2.5-coder:7b"}' 2>/dev/null)

    if echo "$RESPONSE" | grep -q "response"; then
        echo -e "${GREEN}✓ Chat endpoint working${NC}"
    else
        echo -e "${YELLOW}⚠ Chat endpoint may need more time${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping chat test (no API key)${NC}"
fi

# Summary
echo ""
echo "[9/9] Summary"
echo "═══════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ LocalLLM is ready!${NC}"
echo ""
echo "📝 Next Steps:"
echo "  1. Add your GitHub token to .env (for LoRA training)"
echo "  2. Access UI: http://localhost:3001"
echo "  3. Read SETUP_GUIDE.md for detailed instructions"
echo ""
echo "🔑 Your API Key: $API_KEY"
echo ""
echo "📚 Quick Commands:"
echo "  # Test RAG"
echo "  curl -X POST http://localhost:8007/documents/add \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"content\": \"Test document\", \"collection_name\": \"knowledge_base\"}'"
echo ""
echo "  # Chat with RAG"
echo "  curl -X POST http://localhost:8080/chat \\"
echo "    -H 'Authorization: Bearer $API_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"test\", \"model\": \"qwen2.5-coder:7b\", \"use_rag\": true}'"
echo ""
echo "  # Train LoRA profile (after adding GitHub token)"
echo "  cd scripts/lora_profiles"
echo "  python fine_tune_profile_lora.py --profile android \\"
echo "    --dataset ../../datasets/lora_profiles/android_mobile_starter.jsonl \\"
echo "    --epochs 3"
echo ""
echo "═══════════════════════════════════════════════"
