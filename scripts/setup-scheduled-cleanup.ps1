# Setup Automatic Background Process Cleanup
# This script creates a Windows Scheduled Task to run cleanup every 15 minutes

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Automatic Cleanup Scheduler Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please right-click and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

# Configuration
$taskName = "LocalLLM-AutoCleanup"
$scriptPath = "C:\BOT\localLLM\cleanup.bat"
$interval = 15  # minutes

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Task Name: $taskName"
Write-Host "  Script: $scriptPath"
Write-Host "  Interval: Every $interval minutes"
Write-Host ""

# Check if script exists
if (-not (Test-Path $scriptPath)) {
    Write-Host "ERROR: Cleanup script not found at: $scriptPath" -ForegroundColor Red
    Write-Host ""
    pause
    exit 1
}

# Remove existing task if it exists
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "  Existing task removed" -ForegroundColor Green
}

# Create the scheduled task
Write-Host ""
Write-Host "Creating scheduled task..." -ForegroundColor Cyan

# Create action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "C:\BOT\localLLM"

# Create trigger (every 15 minutes, indefinitely)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes $interval) -RepetitionDuration (New-TimeSpan -Days 9999)

# Create settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable:$false `
    -DontStopOnIdleEnd

# Create principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Register the task
$task = Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Automatically clean up LocalLLM background processes every $interval minutes"

if ($task) {
    Write-Host ""
    Write-Host "SUCCESS! Automatic cleanup is now scheduled!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  Name: $taskName"
    Write-Host "  Status: Ready"
    Write-Host "  Runs: Every $interval minutes"
    Write-Host "  Next Run: Within $interval minutes"
    Write-Host ""
    Write-Host "To manage this task:" -ForegroundColor Yellow
    Write-Host "  1. Open Task Scheduler (taskschd.msc)"
    Write-Host "  2. Find '$taskName' in the Task Scheduler Library"
    Write-Host "  3. Right-click to Run, Disable, or Delete"
    Write-Host ""
    Write-Host "To disable automatic cleanup:" -ForegroundColor Yellow
    Write-Host "  Run: Disable-ScheduledTask -TaskName '$taskName'"
    Write-Host ""
    Write-Host "To remove automatic cleanup:" -ForegroundColor Yellow
    Write-Host "  Run: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host ""
}

pause
