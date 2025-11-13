@echo off
REM Automatic Background Process Cleanup
REM Quick cleanup of all background bash processes from Redis

echo.
echo =====================================================
echo   Background Process Cleanup Tool
echo =====================================================
echo.

docker exec localllm-redis redis-cli --scan --pattern "bash:*" | xargs -r -I {} docker exec localllm-redis redis-cli DEL {}

echo.
echo [SUCCESS] All background processes cleaned up!
echo.
echo You can run this anytime with: cleanup.bat
echo.
pause
