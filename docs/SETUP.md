# Setup & Installation

This guide covers setting up Profoot on your local machine for development or personal use.

## Prerequisites

1. **Python 3.10+** — [Download here](https://www.python.org/downloads/)
2. **Tesseract OCR** — Required for scanning image-based/scanned PDFs.
   - **macOS:** `brew install tesseract`
   - **Windows:** [Download binary here](https://github.com/UB-Mannheim/tesseract/wiki)
3. **Google API Key** — Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/profoot.git
cd profoot
```

### 2. Set Up a Virtual Environment

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_primary_key_here

# Optional: Add more keys for automatic rotation if you hit rate limits
GOOGLE_API_KEY_2=your_second_key_here
GOOGLE_API_KEY_3=your_third_key_here
```

## Running the Application

### Via Command Line

```bash
streamlit run app.py
```

### Via Launchers

- **macOS:** Double-click `Start Chatbot (Mac).command`
- **Windows:** Double-click `Start Chatbot (Windows).bat`

## Troubleshooting

- **OCR Errors:** Ensure Tesseract is in your system PATH.
- **SQLite/Chroma Errors:** If the database becomes corrupted, you can safely delete `chat_history.db` or the `chroma_db/` folder; the app will recreate them on next run.
