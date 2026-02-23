# Project Architecture

Profoot is built on a modular RAG (Retrieval-Augmented Generation) pipeline designed specifically for high-accuracy textbook analysis.

## Core Pipeline

### 1. PDF Ingestion & OCR

When a PDF is uploaded, Profoot performs a multi-stage analysis:

- **Chapter Detection:** First, it attempts native text extraction to find "Hoofdstuk" (Chapter) markers. If the PDF is image-only, it runs a high-speed OCR pass on only the top portion of each page to identify headers.
- **Full OCR:** Every page is processed via Tesseract to ensure 100% text coverage, regardless of the PDF source quality.

### 2. Vectorization & Storage

- **Chunking:** Text is split into 800-character segments with a 150-character overlap using `RecursiveCharacterTextSplitter`.
- **Embeddings:** We use `BAAI/bge-small-en-v1.5` via the FastEmbed library for efficient, high-quality local embeddings.
- **Vector DB:** Chunks are stored in a local ChromaDB instance, tagged with chapter and source page metadata.

### 3. Retrieval & Generation

- **Context Filtering:** Users can focus searches on specific chapters. The retriever uses similarity search to pull the top 5 most relevant segments.
- **Prompt Engineering:** The system uses a two-stage prompt strategy. It first summarizes the textbook text, then supplements with general medical knowledge if the textbook content is insufficient.
- **Multi-Model Fallback:** The application rotates through a chain of models (Gemini 2.5 Flash, Gemini 1.5 Pro, etc.) to ensure high availability and bypass individual model rate limits.

## Key Components

- **`app.py`:** Main Streamlit application, UI logic, and session state management.
- **`db_utils.py`:** SQLite handler for chat persistence and quiz history.
- **`test_utils.py`:** Logic for generating proportionally distributed quizzes across textbook chapters.
- **`scripts/build_vector_db.py`:** The backend processing engine for OCR and embedding.

## Data Schema

### SQLite (chat_history.db)

- `sessions`: Stores conversation metadata and titles.
- `messages`: Stores full Q&A history (limited to 20 messages per session for performance).
- `past_questions`: Logs generated quiz questions to ensure variety in future tests.

### ChromaDB (chroma_db/)

- Collection: `langchain`
- Metadata: `{"page": int, "chapter": string, "source": string}`
