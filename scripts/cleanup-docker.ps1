# Docker Cleanup Script for Windows PowerShell
# This script removes unused Docker resources to free up disk space

Write-Host "=== Docker Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""

# Show current disk usage
Write-Host "Current Docker disk usage:" -ForegroundColor Yellow
docker system df

Write-Host ""
Write-Host "Proceeding with cleanup..." -ForegroundColor Yellow

# Remove unused images (not just dangling, but unused images from stopped containers)
Write-Host "`n1. Removing unused images..." -ForegroundColor Green
docker image prune -a -f

# Remove unused containers
Write-Host "`n2. Removing stopped containers..." -ForegroundColor Green
docker container prune -f

# Remove unused volumes (be careful - this removes volumes not used by running containers)
Write-Host "`n3. Removing unused volumes..." -ForegroundColor Green
docker volume prune -f

# Remove unused networks
Write-Host "`n4. Removing unused networks..." -ForegroundColor Green
docker network prune -f

# Remove build cache
Write-Host "`n5. Removing build cache..." -ForegroundColor Green
docker builder prune -a -f

# Show disk usage after cleanup
Write-Host ""
Write-Host "Disk usage after cleanup:" -ForegroundColor Yellow
docker system df

Write-Host ""
Write-Host "=== Cleanup Complete ===" -ForegroundColor Cyan

