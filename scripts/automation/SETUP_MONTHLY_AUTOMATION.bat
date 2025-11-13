@echo off
REM Monthly Fine-Tuning Automation Setup
REM Runs monthly_unsloth_automation.py on the 1st of each month @ 3 AM

echo ================================================================================
echo MONTHLY FINE-TUNING AUTOMATION SETUP
echo ================================================================================
echo.
echo This will create a Windows Task Scheduler task that:
echo - Runs monthly on the 1st @ 3:00 AM
echo - Merges all weekly reflexion data
echo - Prepares datasets for Unsloth training
echo - Creates training report
echo.
echo Prerequisites:
echo 1. Weekly automation already set up (SETUP_WEEKLY_AUTOMATION.bat)
echo 2. Google Drive folder created (lora_datasets)
echo 3. Colab notebook created and bookmarked
echo.
pause

REM Get current directory
set SCRIPT_DIR=%CD%
set PYTHON_EXE=python

echo.
echo Script directory: %SCRIPT_DIR%
echo Python executable: %PYTHON_EXE%
echo.

REM Create task
echo Creating Windows Task Scheduler task...
schtasks /create /tn "LocalLLM Monthly Fine-Tuning" /tr "%PYTHON_EXE% %SCRIPT_DIR%\monthly_unsloth_automation.py" /sc monthly /d 1 /st 03:00 /rl highest /f

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! Monthly automation configured!
    echo ================================================================================
    echo.
    echo Task: "LocalLLM Monthly Fine-Tuning"
    echo Schedule: 1st of each month @ 3:00 AM
    echo Action: Merge data and prepare for training
    echo.
    echo What happens automatically:
    echo 1. Merges all weekly reflexion data
    echo 2. Combines with original starter datasets
    echo 3. Saves to colab_sync/ folder
    echo 4. Creates training report
    echo.
    echo Manual steps after automation runs:
    echo 1. Upload colab_sync/* to Google Drive
    echo 2. Open Colab notebook
    echo 3. Click "Run all"
    echo 4. Wait 3 hours ^(or 36 min with Pro^)
    echo 5. Download and deploy models
    echo.
    echo Or: Use Colab Pro scheduled execution for 100%% automation!
    echo See: COLAB_AUTOMATION_GUIDE.md
    echo.
    echo To test now: python monthly_unsloth_automation.py
    echo.
) else (
    echo.
    echo ERROR: Failed to create task!
    echo Please run this script as Administrator.
    echo.
)

pause
