@echo off
REM Security check before pushing to GitHub
REM Run this to ensure no sensitive data will be exposed

echo ====================================
echo LocalLLM Security Check
echo ====================================
echo.

echo [1/4] Checking for Anthropic API keys...
findstr /R /C:"sk-ant-" /S *.py *.js *.md *.bat *.sh 2>nul
if %errorlevel% equ 0 (
    echo [ERROR] Found Anthropic API keys in files!
    echo Please remove them before pushing to GitHub.
    goto :error
) else (
    echo [OK] No Anthropic API keys found
)
echo.

echo [2/4] Checking for GitHub tokens...
findstr /R /C:"github_pat_" /S *.py *.js *.md *.bat *.sh *.env 2>nul
if %errorlevel% equ 0 (
    echo [ERROR] Found GitHub tokens in files!
    echo Please remove them before pushing to GitHub.
    goto :error
) else (
    echo [OK] No GitHub tokens found
)
echo.

echo [3/4] Checking .env file...
findstr /C:"your_github_token_here" .env >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] .env might contain a real token
    echo Make sure .env has placeholder: your_github_token_here
) else (
    echo [OK] .env has placeholder token
)
echo.

echo [4/4] Checking .gitignore...
findstr /C:".env" .gitignore >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] .env is in .gitignore
) else (
    echo [ERROR] .env is NOT in .gitignore!
    goto :error
)
echo.

echo ====================================
echo Security Check PASSED
echo ====================================
echo.
echo Your repository is safe to push to GitHub!
echo.
echo Next steps:
echo 1. Rename README: move README.md README_INTERNAL.md
echo 2. Use GitHub README: move README_GITHUB.md README.md
echo 3. Push to GitHub: git push origin main
echo.
exit /b 0

:error
echo.
echo ====================================
echo Security Check FAILED
echo ====================================
echo.
echo DO NOT push to GitHub until you fix the issues above!
echo.
exit /b 1
