@echo off
echo Starting Dutch Anatomy Chatbot Setup...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in the system PATH. Please install Python first.
    pause
    exit /b
)

:: Check if Tesseract is installed (Required for Windows OCR if they ever want to rebuild the DB)
:: Note: The DB is already built and provided in chroma_db/ so they don't NEED it unless they run build_vector_db.py again.
echo Checking for virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Installing dependencies...
    .venv\Scripts\pip install -r requirements.txt
) else (
    echo Virtual environment found.
)

echo Starting the Web App...
.venv\Scripts\streamlit run app.py
pause
