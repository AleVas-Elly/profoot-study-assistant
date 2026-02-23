# Profoot — AI Study Assistant

Profoot is a self-hosted AI study assistant that turns any PDF textbook into an interactive learning tool. Upload a book, and you can chat with it or generate a timed multiple-choice quiz — all powered by Google Gemini and a local vector database.

## Features

- **Study Mode** — Ask free-form questions and get detailed, source-cited answers drawn directly from your textbook.
- **Test Mode** — Generate a personalised multiple-choice quiz from any chapter or the full book, complete with explanations for correct and incorrect answers.
- **Smart PDF Ingestion** — Automatically detects chapter boundaries in both born-digital and scanned (image-based) PDFs using OCR.
- **Multi-key API Rotation** — Distributes requests across multiple Google API keys and rotates automatically when rate limits are hit.
- **Persistent Chat History** — Conversations are stored locally in SQLite so you can pick up where you left off.

## Tech Stack

| Layer        | Technology                                                                                           |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| Frontend     | [Streamlit](https://streamlit.io/)                                                                   |
| Embeddings   | [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`)                          |
| Vector DB    | [ChromaDB](https://www.trychroma.com/)                                                               |
| LLM          | Google Gemini via [LangChain](https://python.langchain.com/)                                         |
| OCR          | [PyMuPDF](https://pymupdf.readthedocs.io/) + [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| Chat History | SQLite                                                                                               |

## Architecture Overview

```
PDF Upload
    │
    ▼
Chapter Detection (PyMuPDF native text / Tesseract OCR top-strip)
    │
    ▼
Full OCR  →  Text Chunking  →  FastEmbed  →  ChromaDB
                                                  │
               User Question ─────────────────────┤
                                                  │
                                           Vector Search
                                                  │
                                           Gemini LLM
                                                  │
                                          Streamed Answer
```

## Quick Start

See **[docs/SETUP.md](docs/SETUP.md)** for full installation instructions.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/profoot.git
cd profoot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Google API key to a .env file
echo "GOOGLE_API_KEY=your_key_here" > .env

# 5. Launch the app
streamlit run app.py
```

> One-click launchers for Mac (`Start Chatbot (Mac).command`) and Windows (`Start Chatbot (Windows).bat`) are also included.

## Documentation

| Document                                     | Description                                                       |
| -------------------------------------------- | ----------------------------------------------------------------- |
| [docs/SETUP.md](docs/SETUP.md)               | Prerequisites, installation, and environment configuration        |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Deep-dive into the RAG pipeline, chapter detection, and data flow |
| [docs/USAGE.md](docs/USAGE.md)               | How to use Study Mode, Test Mode, and the PDF uploader            |

## License

MIT
