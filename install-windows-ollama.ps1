# Windows Ollama Installation Script
# Run this in Windows PowerShell as Administrator

Write-Host "=== Windows Ollama Installation Script ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Ollama is already installed
Write-Host "Step 1: Checking if Ollama is already installed..." -ForegroundColor Yellow
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaPath) {
    Write-Host "✓ Ollama is already installed at: $($ollamaPath.Source)" -ForegroundColor Green
    $ollamaVersion = & ollama --version
    Write-Host "  Version: $ollamaVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Ollama not found. Installing..." -ForegroundColor Red

    # Try winget first
    Write-Host "Attempting installation via winget..." -ForegroundColor Yellow
    try {
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
        Write-Host "✓ Ollama installed successfully via winget!" -ForegroundColor Green
    } catch {
        Write-Host "✗ winget failed. Please download manually from: https://ollama.com/download/windows" -ForegroundColor Red
        Write-Host "Press any key to open download page..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        Start-Process "https://ollama.com/download/windows"
        Write-Host "After installation, run this script again." -ForegroundColor Yellow
        exit
    }
}

# Step 2: Configure Ollama for network access
Write-Host ""
Write-Host "Step 2: Configuring Ollama for network access..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'User')
Write-Host "✓ Environment variable OLLAMA_HOST set to 0.0.0.0:11434" -ForegroundColor Green

# Step 3: Restart Ollama service
Write-Host ""
Write-Host "Step 3: Restarting Ollama service..." -ForegroundColor Yellow
try {
    Stop-Service Ollama -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Start-Service Ollama -ErrorAction SilentlyContinue
    Write-Host "✓ Ollama service restarted" -ForegroundColor Green
} catch {
    Write-Host "⚠ Service restart failed. You may need to restart manually or reboot Windows." -ForegroundColor Yellow
}

# Step 4: Pull models
Write-Host ""
Write-Host "Step 4: Pulling models (this may take 10-15 minutes)..." -ForegroundColor Yellow
Write-Host "Pulling qwen3:latest (5.2GB)..." -ForegroundColor Cyan
& ollama pull qwen3:latest

Write-Host "Pulling deepseek-r1:7b (4.7GB)..." -ForegroundColor Cyan
& ollama pull deepseek-r1:7b

Write-Host "Pulling qwen3-coder:latest (18GB - large!)..." -ForegroundColor Cyan
& ollama pull qwen3-coder:latest

# Step 5: List installed models
Write-Host ""
Write-Host "Step 5: Verifying installed models..." -ForegroundColor Yellow
& ollama list

# Step 6: Test GPU acceleration
Write-Host ""
Write-Host "Step 6: Testing GPU acceleration..." -ForegroundColor Yellow
Write-Host "Running test query... (Watch Task Manager > Performance > GPU)" -ForegroundColor Cyan
& ollama run qwen3:latest "Say hello in one sentence"

# Step 7: Configure firewall
Write-Host ""
Write-Host "Step 7: Configuring Windows Firewall..." -ForegroundColor Yellow
try {
    New-NetFirewallRule -DisplayName "Ollama WSL2 Access" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow -ErrorAction SilentlyContinue
    Write-Host "✓ Firewall rule created for port 11434" -ForegroundColor Green
} catch {
    Write-Host "⚠ Firewall rule may already exist or requires admin privileges" -ForegroundColor Yellow
}

# Done
Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Open Task Manager > Performance > GPU to verify GPU usage during inference"
Write-Host "2. Run: docker-compose -f docker-compose-simplified.yml stop ollama"
Write-Host "3. Update docker-compose.yml to use host.docker.internal:11434"
Write-Host "4. Restart Docker services: docker-compose -f docker-compose-simplified.yml up -d"
Write-Host ""
Write-Host "Test from WSL2: curl http://localhost:11434/api/tags" -ForegroundColor Yellow
