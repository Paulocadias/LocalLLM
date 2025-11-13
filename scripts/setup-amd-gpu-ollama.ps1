# AMD GPU Setup Script for Ollama on Windows
# This script configures Ollama to use AMD RX6900 GPU

Write-Host "Setting up AMD GPU acceleration for Ollama..." -ForegroundColor Green

# Set environment variables for AMD ROCm
Write-Host "Setting environment variables..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable('OLLAMA_GPU_DRIVER', 'rocm', 'User')
[System.Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES', '0', 'User')
[System.Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', '10.3.0', 'User')

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  OLLAMA_GPU_DRIVER=rocm" -ForegroundColor Cyan
Write-Host "  HIP_VISIBLE_DEVICES=0" -ForegroundColor Cyan
Write-Host "  HSA_OVERRIDE_GFX_VERSION=10.3.0" -ForegroundColor Cyan

# Stop any running Ollama processes
Write-Host "Stopping existing Ollama processes..." -ForegroundColor Yellow
Stop-Process -Name '*ollama*' -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Start Ollama with GPU support
Write-Host "Starting Ollama with GPU support..." -ForegroundColor Yellow
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (Test-Path $ollamaPath) {
    Start-Process $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden
    Write-Host "Ollama started with GPU support" -ForegroundColor Green
} else {
    Write-Host "Error: Ollama not found at $ollamaPath" -ForegroundColor Red
    exit 1
}

Write-Host "Waiting for Ollama to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test if Ollama is running
Write-Host "Testing Ollama service..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -Method Get
    Write-Host "Ollama is running successfully!" -ForegroundColor Green
    Write-Host "Available models:" -ForegroundColor Cyan
    $response.models | ForEach-Object { Write-Host "  - $($_.name)" -ForegroundColor White }
} catch {
    Write-Host "Error: Ollama service is not responding" -ForegroundColor Red
    Write-Host "Please check if Ollama is installed correctly" -ForegroundColor Yellow
}

Write-Host "`nAMD GPU Setup Complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Test model inference: ollama run qwen3:latest" -ForegroundColor Cyan
Write-Host "2. Check GPU usage in Task Manager > Performance > GPU" -ForegroundColor Cyan
Write-Host "3. Monitor VRAM usage during model inference" -ForegroundColor Cyan

Write-Host "`nNote: For optimal performance, ensure you have:" -ForegroundColor Yellow
Write-Host "- Latest AMD Adrenalin drivers installed" -ForegroundColor White
Write-Host "- Windows 11 with WSL2 for ROCm support" -ForegroundColor White
Write-Host "- Sufficient VRAM (RX6900 has 16GB)" -ForegroundColor White
