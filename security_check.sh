#!/bin/bash
# Security check before pushing to GitHub
# Run this to ensure no sensitive data will be exposed

echo "===================================="
echo "LocalLLM Security Check"
echo "===================================="
echo ""

echo "[1/4] Checking for Anthropic API keys..."
if grep -r "sk-ant-" --include="*.py" --include="*.js" --include="*.md" --include="*.sh" --include="*.bat" . 2>/dev/null | grep -v ".git"; then
    echo "[ERROR] Found Anthropic API keys in files!"
    echo "Please remove them before pushing to GitHub."
    exit 1
else
    echo "[OK] No Anthropic API keys found"
fi
echo ""

echo "[2/4] Checking for GitHub tokens..."
if grep -r "github_pat_" --include="*.py" --include="*.js" --include="*.md" --include="*.sh" --include="*.bat" --include="*.env" . 2>/dev/null | grep -v ".git"; then
    echo "[ERROR] Found GitHub tokens in files!"
    echo "Please remove them before pushing to GitHub."
    exit 1
else
    echo "[OK] No GitHub tokens found"
fi
echo ""

echo "[3/4] Checking .env file..."
if grep -q "your_github_token_here" .env 2>/dev/null; then
    echo "[OK] .env has placeholder token"
else
    echo "[WARNING] .env might contain a real token"
    echo "Make sure .env has placeholder: your_github_token_here"
fi
echo ""

echo "[4/4] Checking .gitignore..."
if grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo "[OK] .env is in .gitignore"
else
    echo "[ERROR] .env is NOT in .gitignore!"
    exit 1
fi
echo ""

echo "===================================="
echo "Security Check PASSED"
echo "===================================="
echo ""
echo "Your repository is safe to push to GitHub!"
echo ""
echo "Next steps:"
echo "1. Rename README: mv README.md README_INTERNAL.md"
echo "2. Use GitHub README: mv README_GITHUB.md README.md"
echo "3. Push to GitHub: git push origin main"
echo ""
exit 0
