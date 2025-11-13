# Monitor LoRA Training on Windows PowerShell

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  LoRA Training Monitor" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$logFile = "quick_training_32b.log"

# Check if log exists
if (-Not (Test-Path $logFile)) {
    Write-Host "Log file not found: $logFile" -ForegroundColor Red
    Write-Host "Training may not have started yet." -ForegroundColor Yellow
    exit
}

# Get file size
$size = (Get-Item $logFile).Length / 1MB
Write-Host "Log file size: $([math]::Round($size, 2)) MB" -ForegroundColor Green
Write-Host ""

# Check which profile is training
Write-Host "Current Status:" -ForegroundColor Yellow
$currentProfile = Select-String -Path $logFile -Pattern "Training.*profile" | Select-Object -Last 1
if ($currentProfile) {
    Write-Host $currentProfile.Line -ForegroundColor Green
} else {
    Write-Host "Downloading model..." -ForegroundColor Yellow
}
Write-Host ""

# Check for completion
$complete = Select-String -Path $logFile -Pattern "All training complete"
if ($complete) {
    Write-Host "✅ TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. cd ../.."
    Write-Host "  2. bash scripts/lora_profiles/create_modelfiles.sh"
    Write-Host "  3. bash scripts/lora_profiles/register_models.sh"
    exit
}

# Show last 30 lines
Write-Host "Last 30 lines:" -ForegroundColor Cyan
Write-Host "--------------------------------------"
Get-Content $logFile -Tail 30
Write-Host "--------------------------------------"
Write-Host ""
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
Write-Host "Refreshing every 10 seconds..." -ForegroundColor Yellow

# Auto-refresh loop
while ($true) {
    Start-Sleep -Seconds 10
    Clear-Host
    & $MyInvocation.MyCommand.Path
    break
}
