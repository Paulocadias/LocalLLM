# Cleanup aibrain containers and images
# This will stop and remove all aibrain containers and their images

Write-Host "=== Aibrain Cleanup Script ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "WARNING: This will stop and remove all aibrain containers and images!" -ForegroundColor Red
Write-Host ""

# Get all aibrain containers
$containers = docker ps -a --filter "name=aibrain" --format "{{.Names}}"

if ($containers) {
    Write-Host "Found aibrain containers:" -ForegroundColor Yellow
    $containers
    
    Write-Host "`nStopping containers..." -ForegroundColor Green
    docker stop $containers
    
    Write-Host "`nRemoving containers..." -ForegroundColor Green
    docker rm $containers
    
    Write-Host "`nRemoving images..." -ForegroundColor Green
    docker images --filter "reference=aibrain*" --format "{{.ID}}" | ForEach-Object {
        docker rmi $_ -f
    }
    
    Write-Host "`nCleanup complete!" -ForegroundColor Green
} else {
    Write-Host "No aibrain containers found." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Disk usage after cleanup:" -ForegroundColor Yellow
docker system df

