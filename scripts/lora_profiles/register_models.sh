#!/bin/bash
# Register LoRA models with Ollama

cd ../..

echo "============================================"
echo "Registering LoRA Models with Ollama"
echo "============================================"
echo ""

# Android Profile
echo "[1/6] Registering qwen-android-mobile..."
ollama create qwen-android-mobile -f Modelfile.android
if [ $? -eq 0 ]; then
    echo "✅ qwen-android-mobile registered successfully"
else
    echo "❌ Failed to register qwen-android-mobile"
fi
echo ""

# Backend Profile
echo "[2/6] Registering qwen-backend..."
ollama create qwen-backend -f Modelfile.backend
if [ $? -eq 0 ]; then
    echo "✅ qwen-backend registered successfully"
else
    echo "❌ Failed to register qwen-backend"
fi
echo ""

# Frontend Profile
echo "[3/6] Registering qwen-frontend..."
ollama create qwen-frontend -f Modelfile.frontend
if [ $? -eq 0 ]; then
    echo "✅ qwen-frontend registered successfully"
else
    echo "❌ Failed to register qwen-frontend"
fi
echo ""

# Career Advisor Profile
echo "[4/6] Registering qwen-career-advisor..."
ollama create qwen-career-advisor -f Modelfile.career
if [ $? -eq 0 ]; then
    echo "✅ qwen-career-advisor registered successfully"
else
    echo "❌ Failed to register qwen-career-advisor"
fi
echo ""

# Marketing Specialist Profile
echo "[5/6] Registering qwen-marketing..."
ollama create qwen-marketing -f Modelfile.marketing
if [ $? -eq 0 ]; then
    echo "✅ qwen-marketing registered successfully"
else
    echo "❌ Failed to register qwen-marketing"
fi
echo ""

# Website Builder Profile
echo "[6/6] Registering qwen-website..."
ollama create qwen-website -f Modelfile.website
if [ $? -eq 0 ]; then
    echo "✅ qwen-website registered successfully"
else
    echo "❌ Failed to register qwen-website"
fi
echo ""

echo "============================================"
echo "Verification"
echo "============================================"
ollama list | grep qwen

echo ""
echo "✅ All LoRA models registered!"
echo ""
echo "Coding Profiles:"
echo "  ollama run qwen-android-mobile"
echo "  ollama run qwen-backend"
echo "  ollama run qwen-frontend"
echo ""
echo "Business & Consulting Profiles:"
echo "  ollama run qwen-career-advisor"
echo "  ollama run qwen-marketing"
echo "  ollama run qwen-website"
