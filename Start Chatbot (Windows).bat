@echo off
setlocal enabledelayedexpansion
title Profoot - Windows Launcher

:: Navigate to the folder where this script is located
cd /d "%~dp0"

echo ===================================================
echo     Profoot - Windows Launcher
echo ===================================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo.
    echo Please follow these steps:
    echo 1. Go to https://www.python.org/downloads/
    echo 2. Download the latest Python installer for Windows.
    echo 3. Run the installer.
    echo 4. IMPORTANT: Check the box "Add Python to PATH" at the bottom of the installer window!
    echo 5. Click "Install Now".
    echo 6. Once installed, run this file again.
    echo.
    pause
    exit /b 1
)

:: 2. Check if virtual environment exists, create if not
if not exist ".venv_win\Scripts\activate.bat" (
    echo [INFO] First time setup: Creating Python virtual environment...
    python -m venv .venv_win
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. 
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

:: 3. Activate the environment
call .venv_win\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Install dependencies
echo [INFO] Checking dependencies. This might take a few minutes on the first run...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Please check your internet connection.
    pause
    exit /b 1
)

:: 5. Run the app
echo [INFO] Setup complete! Launching the Application...
echo [INFO] Your browser should open automatically. Do not close this black window while using the app.
python -m streamlit run app.py

:: Keep window open if the app crashes or is closed
echo.
echo Application stopped.
pause
