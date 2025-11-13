@echo off
REM ============================================================================
REM Setup Weekly Automatic Improvement System
REM
REM This creates a Windows Task Scheduler task that runs every Monday at 2 AM
REM "Set and Forget" - runs automatically!
REM ============================================================================

echo.
echo ================================================================================
echo SETUP WEEKLY AUTOMATIC IMPROVEMENT
echo ================================================================================
echo.
echo This will create a Windows Task Scheduler task that:
echo   - Runs every Monday at 2 AM
echo   - Tests all models vs Claude
echo   - Generates reflexion training data
echo   - Creates improved models
echo   - Generates weekly reports
echo.
echo You need to run this as Administrator!
echo.
pause

REM Get current directory
set "CURRENT_DIR=%CD%"
set "PYTHON_EXE=%CURRENT_DIR%\venv\Scripts\python.exe"
set "SCRIPT_PATH=%CURRENT_DIR%\weekly_complete_improvement.py"

REM Check if Python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found at %PYTHON_EXE%
    echo Please update PYTHON_EXE in this script
    echo.
    pause
    exit /b 1
)

REM Check if script exists
if not exist "%SCRIPT_PATH%" (
    echo ERROR: Script not found at %SCRIPT_PATH%
    echo.
    pause
    exit /b 1
)

echo.
echo Creating Windows Task Scheduler task...
echo.

REM Create the task
schtasks /create /tn "LocalLLM_Weekly_Improvement" /tr "\"%PYTHON_EXE%\" \"%SCRIPT_PATH%\"" /sc weekly /d MON /st 02:00 /f /rl highest

if %ERRORLEVEL% == 0 (
    echo.
    echo ================================================================================
    echo [SUCCESS] Weekly automation task created!
    echo ================================================================================
    echo.
    echo Task Details:
    echo   Name: LocalLLM_Weekly_Improvement
    echo   Schedule: Every Monday at 2:00 AM
    echo   Script: %SCRIPT_PATH%
    echo   Python: %PYTHON_EXE%
    echo.
    echo The task will run automatically every week!
    echo.
    echo To view the task:
    echo   - Open Task Scheduler
    echo   - Look for "LocalLLM_Weekly_Improvement"
    echo.
    echo To run it manually:
    echo   schtasks /run /tn "LocalLLM_Weekly_Improvement"
    echo.
    echo To delete the task:
    echo   schtasks /delete /tn "LocalLLM_Weekly_Improvement" /f
    echo.
    echo ================================================================================
) else (
    echo.
    echo [ERROR] Failed to create task!
    echo.
    echo Make sure you:
    echo   1. Run this script as Administrator
    echo   2. Have Task Scheduler service enabled
    echo.
)

echo.
pause
