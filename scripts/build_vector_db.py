import os
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma_db"


def detect_chapters(pdf_file_or_path):
    """
    Detects chapter headings in a PDF.
    For born-digital PDFs, uses native text extraction. For scanned/image-based PDFs,
    falls back to a lightweight OCR pass on the top 15% of each page.
    Returns a list of {"name": ..., "start_page": ...} dicts sorted by page (1-indexed).
    """
    if isinstance(pdf_file_or_path, str):
        doc = fitz.open(pdf_file_or_path)
    else:
        pdf_bytes = pdf_file_or_path.read()
        pdf_file_or_path.seek(0)  # reset for the later full OCR pass
        doc = fitz.open("pdf", pdf_bytes)

    # Only match lines that start with 'Hoofdstuk'
    CHAPTER_KEYWORDS = re.compile(
        r'^hoofdstuk',
        re.IGNORECASE
    )

    total_pages = len(doc)

    # Pass 1: attempt native text extraction
    native_text_found = False
    font_size_counts = {}
    all_spans = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text, line_size = "", 0
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    s = round(span.get("size", 0), 1)
                    if t:
                        line_text += t + " "
                        line_size = max(line_size, s)
                        font_size_counts[s] = font_size_counts.get(s, 0) + len(t)
                        native_text_found = True
                line_text = line_text.strip()
                if line_text and line_size > 0:
                    all_spans.append((page_num, line_size, line_text))

    # Pass 2: image-based PDF — OCR the top strip of each page
    if not native_text_found:
        print("[detect_chapters] Image-based PDF detected — running top-strip OCR scan...")
        ocr_chapters = []
        seen_pages = set()
        seen_names = set()

        for page_idx in tqdm(range(len(doc)), desc="Quick Chapter Scan"):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect

            # Only crop the top 20% of the page — headings live here
            top_strip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.20)
            clip_page = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=top_strip, alpha=False)
            img = Image.open(io.BytesIO(clip_page.tobytes("jpeg")))
            text = pytesseract.image_to_string(img, lang='nld').strip()

            if not text:
                continue

            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in lines:
                # Must START with Hoofdstuk
                has_keyword = bool(CHAPTER_KEYWORDS.match(line))
                is_short = len(line) <= 60
                is_not_number = not re.match(r'^\d+[\s\.]*$', line)

                if has_keyword and is_short and is_not_number and page_num not in seen_pages:
                    # Extract only "Hoofdstuk X" — drop any trailing noise
                    m = re.match(r'(hoofdstuk\s*\|?\s*\d+)', line, re.IGNORECASE)
                    clean = re.sub(r'\s+', ' ', m.group(1)).strip().title() if m else re.sub(r'\s+', ' ', line).strip()
                    if clean not in seen_names:
                        seen_names.add(clean)
                        seen_pages.add(page_num)
                        ocr_chapters.append({"name": clean, "start_page": page_num})
                    break

        if not ocr_chapters:
            ocr_chapters = [{"name": "Full Book", "start_page": 1}]
        else:
            # Deduplicate: keep only the FIRST occurrence of each chapter number/type.
            # Many textbooks print the chapter title as a running header on every page.
            # We detect this by normalising the name and grouping by chapter keyword + number.
            def _chapter_key(name):
                """Return a canonical key like 'hoofdstuk_3' or fallback to lowercased name."""
                m = re.search(r'(hoofdstuk|chapter|deel|unit)\D{0,3}(\d+)', name, re.IGNORECASE)
                if m:
                    return m.group(1).lower() + "_" + m.group(2)
                return name.lower()[:20]

            deduped = []
            seen_keys = set()
            for ch in ocr_chapters:
                k = _chapter_key(ch["name"])
                if k not in seen_keys:
                    seen_keys.add(k)
                    deduped.append(ch)
            ocr_chapters = deduped

        return sorted(ocr_chapters, key=lambda x: x["start_page"]), total_pages

    # Native text path: find headings by font size
    if not font_size_counts:
        return [{"name": "Full Book", "start_page": 1}], total_pages

    body_size = max(font_size_counts, key=font_size_counts.get)
    heading_threshold = body_size * 1.25

    chapters = []
    seen_pages_n = set()
    seen_names_n = set()

    for page_num, font_size, text in all_spans:
        if page_num in seen_pages_n:
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text or len(text) > 80:
            continue
        is_large = font_size >= heading_threshold
        has_keyword = bool(CHAPTER_KEYWORDS.search(text))
        is_short = len(text) <= 70
        is_not_num = not re.match(r'^\d+[\.\s]*$', text)

        if (is_large or has_keyword) and is_short and is_not_num:
            if text not in seen_names_n:
                seen_names_n.add(text)
                seen_pages_n.add(page_num)
                chapters.append({"name": text, "start_page": page_num})

    if not chapters:
        chapters = [{"name": "Full Book", "start_page": 1}]

    chapters = sorted(chapters, key=lambda x: x["start_page"])

    # Ensure page 1 always has an entry
    if chapters[0]["start_page"] != 1:
        chapters.insert(0, {"name": "Preface / Intro", "start_page": 1})

    return chapters, total_pages


def process_pdf(pdf_file_or_path=None, progress_callback=None, source="book.pdf", chapter_map=None):
    """
    OCRs every page of a PDF and tags each chunk with the correct chapter.

    Args:
        pdf_file_or_path: File path (str) or file-like object.
        progress_callback: Optional callback(pct, message).
        source: Filename/identifier embedded in chunk metadata.
        chapter_map: {page_num: chapter_name}. Defaults to "Unknown Chapter" if None.
    """
    if pdf_file_or_path is None:
        pdf_file_or_path = f"books/{source}"

    print(f"Opening PDF for OCR: {source}")
    if isinstance(pdf_file_or_path, str):
        doc = fitz.open(pdf_file_or_path)
    else:
        doc = fitz.open("pdf", pdf_file_or_path.read())

    if chapter_map is None:
        chapter_map = {1: "Unknown Chapter"}

    documents = []
    current_chapter = next(iter(chapter_map.values()))  # Start with first chapter
    total_pages = len(doc)

    print(f"Processing {total_pages} pages...")
    for i in tqdm(range(total_pages), desc="OCR Pages"):
        if progress_callback:
            progress_callback(i / total_pages * 0.80, f"📖 OCRing Page {i+1} of {total_pages}...")

        page_num = i + 1  # 1-indexed

        # Update chapter if we've hit a boundary
        if page_num in chapter_map:
            current_chapter = chapter_map[page_num]
            print(f"\n[INFO] Entered '{current_chapter}' on page {page_num}")

        page = doc[i]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
        text = pytesseract.image_to_string(img, lang='nld')

        # Clean up text
        text = text.replace('\n\n', ' [PARAGRAPH_BREAK] ').replace('\n', ' ')
        text = text.replace(' [PARAGRAPH_BREAK] ', '\n\n')

        if len(text.strip()) > 50:
            doc_obj = Document(
                page_content=text,
                metadata={"page": page_num, "chapter": current_chapter, "source": source}
            )
            documents.append(doc_obj)

    return documents


def build_vector_db(pdf_file_or_path=None, progress_callback=None, source="book.pdf", chapter_map=None):
    """
    Builds or appends to the ChromaDB vector database for a given PDF.

    Args:
        pdf_file_or_path: File path or file-like object.
        progress_callback: callback(pct, message).
        source: PDF filename used in metadata.
        chapter_map: {page_num: chapter_name} from the chapter editor.
    """
    if pdf_file_or_path is None and os.path.exists(CHROMA_PATH):
        print(f"Vector Database already exists. Skipping rebuild.")
        return

    documents = process_pdf(pdf_file_or_path, progress_callback, source=source, chapter_map=chapter_map)

    if progress_callback: progress_callback(0.82, "✂️ Splitting text into chunks...")
    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages.")

    if progress_callback: progress_callback(0.85, "🧠 Loading Embedding Model...")
    print("\nLoading FastEmbed Embeddings...")
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if progress_callback: progress_callback(0.90, "💾 Embedding & Saving to Database... (this may take a minute)")
    print("\nSaving to ChromaDB...")
    try:
        if os.path.exists(CHROMA_PATH) and pdf_file_or_path is not None:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name="langchain")
            db.add_documents(chunks)
            print(f"Appended {len(chunks)} new chunks to existing database.")
        else:
            db = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory=CHROMA_PATH,
                collection_name="langchain"
            )
            print(f"Created new database with {len(chunks)} chunks.")
    except Exception as e:
        print(f"ERROR: Failed to save to ChromaDB: {e}")
        if "tenants" in str(e).lower() or "no such table" in str(e).lower():
            print("Detected schema corruption. Suggestion: Delete 'chroma_db' folder and try again.")
        raise e

    if progress_callback: progress_callback(1.0, "✅ Database Ready!")


if __name__ == "__main__":
    build_vector_db()
