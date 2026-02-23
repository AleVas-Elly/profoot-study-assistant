import os
import re
import time
import traceback
import uuid
import streamlit as st
import db_utils
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()
CHROMA_PATH = "chroma_db"

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

SELECTION_FILE = os.path.join("books", "selection.txt")

def save_selection(book_name):
    if not os.path.exists("books"):
        os.makedirs("books")
    with open(SELECTION_FILE, "w") as f:
        f.write(book_name if book_name else "")

def load_selection():
    if os.path.exists(SELECTION_FILE):
        with open(SELECTION_FILE, "r") as f:
            return f.read().strip() or None
    return None

def get_all_api_keys():
    """Returns all Google API keys found in the environment, deduplicated."""
    keys = []
    primary = os.environ.get("GOOGLE_API_KEY") or st.session_state.get("GOOGLE_API_KEY")
    if primary:
        keys.append(primary)
    for k, v in os.environ.items():
        if k.startswith("GOOGLE_API_KEY_") and v.strip():
            keys.append(v.strip())
    return list(dict.fromkeys(keys))

@st.cache_resource(show_spinner="Loading Database...")
def load_db():
    """Loads the ChromaDB vector store."""
    try:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name="langchain")
    except Exception as e:
        print(f"Failed to load ChromaDB: {e}")
        return None

def delete_book_data(book_filename):
    """Removes a book's PDF, vector embeddings, and question history."""
    try:
        p = os.path.join("books", book_filename)
        if os.path.exists(p):
            os.remove(p)
    except:
        pass

    try:
        db = load_db()
        if db:
            db.delete(where={"source": book_filename})
    except:
        try:
            import chromadb
            client = chromadb.PersistentClient(path="chroma_db")
            coll = client.get_collection("langchain")
            coll.delete(where={"source": book_filename})
        except:
            pass

    try:
        db_utils.delete_past_questions_by_source(book_filename)
    except:
        pass

@st.cache_resource
def load_llms(api_key=None):
    """Initializes the Gemini LLM chain with model fallbacks."""
    target_key = api_key or os.environ.get("GOOGLE_API_KEY") or st.session_state.get("GOOGLE_API_KEY")
    primary   = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=target_key, temperature=0.2, timeout=60.0, max_retries=0)
    fallback_1 = ChatGoogleGenerativeAI(model="gemini-2.5-pro",  google_api_key=target_key, temperature=0.2, timeout=60.0, max_retries=0)
    fallback_2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=target_key, temperature=0.2, timeout=60.0, max_retries=0)
    fallback_3 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=target_key, temperature=0.2, timeout=60.0, max_retries=0)
    return [primary, fallback_1, fallback_2, fallback_3]

def apply_global_styles():
    """Injects global CSS for the glassmorphic theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── GLOBAL GRADIENT BACKGROUND ── */
        .stApp {
            background: linear-gradient(135deg,
                #c9b8f0 0%,
                #f7c5b0 48%,
                #aee4c8 100%
            ) !important;
            background-attachment: fixed !important;
        }

        /* ── TRANSPARENCY OVERRIDES ── */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main,
        .main > div,
        div.block-container,
        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] { background: transparent !important; }

        .block-container {
            padding-top: 1.5rem !important;
            max-width: 850px !important;
        }

        /* ── SIDEBAR GLASSMORPHISM ── */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.4) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.3) !important;
        }
        [data-testid="stSidebarNav"] {
            background-color: transparent !important;
        }
        
        /* ── SIDEBAR TOGGLE FIX ── */
        [data-testid="stSidebarCollapseButton"] {
            background-color: rgba(255, 255, 255, 0.6) !important;
            color: #1e293b !important;
            border-radius: 50% !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            z-index: 10001 !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover {
            background-color: rgba(255, 255, 255, 0.9) !important;
        }

        /* ── TYPOGRAPHY ── */
        h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown, [data-testid="stText"] {
            font-family: 'Inter', sans-serif !important;
            color: #1e293b !important;
        }
        
        /* Exception for primary buttons */
        .stButton > button[kind="primary"] {
            color: #ffffff !important;
        }
        h1, h2, h3 {
            font-weight: 800 !important;
            letter-spacing: -0.04em !important;
        }
        h1 { font-size: 3.5rem !important; margin-bottom: 1.5rem !important; }
        
        /* ── BUTTONS ── */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 14px !important;
            letter-spacing: 0.01em !important;
            width: 100% !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            border: 1px solid rgba(255,255,255,0.4) !important;
        }
        .stButton > button[kind="primary"] {
            background: rgba(59, 130, 246, 0.9) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 15px rgba(59,130,246,0.25) !important;
            border: none !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #2563eb !important;
            transform: translateY(-1.5px) !important;
            box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
        }
        .stButton > button[kind="secondary"] {
            background: rgba(255,255,255,0.35) !important;
            color: #1e293b !important;
            backdrop-filter: blur(10px);
        }
        .stButton > button[kind="secondary"]:hover {
            background: rgba(255,255,255,0.5) !important;
            border-color: rgba(255,255,255,0.8) !important;
        }

        /* ── FORMS & INPUTS ── */
        .stTextInput input, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] {
            background: rgba(255,255,255,0.4) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.5) !important;
            backdrop-filter: blur(12px) !important;
            font-weight: 500 !important;
            color: #1e293b !important;
        }
        
        /* ── CHAT INPUT (Bottom Part) ── */
        [data-testid="stChatInputContainer"] {
            background-color: rgba(255, 255, 255, 0.45) !important;
            backdrop-filter: blur(28px) saturate(1.8) !important;
            -webkit-backdrop-filter: blur(28px) saturate(1.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.7) !important;
            border-radius: 20px !important;
            padding: 10px !important;
            box-shadow: 0 -4px 30px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stChatInputContainer"] textarea {
            background-color: transparent !important;
            color: #1e293b !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* ── GLASS CARD ── */
        .glass-card {
            background: rgba(255, 255, 255, 0.45);
            backdrop-filter: blur(28px) saturate(1.8);
            -webkit-backdrop-filter: blur(28px) saturate(1.8);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
            transition: transform 0.3s ease;
        }

        /* ── CHAT BUBBLES & EMOJIS ── */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            margin-bottom: 1.5rem !important;
            padding: 0 !important;
            display: flex !important;
            align-items: flex-start !important;
        }
        
        /* Avatar/Emoji styling */
        [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarUser"] {
            background-color: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        }

        [data-testid="stChatMessage"] > div[data-testid="stFormatContainer"] {
            border-radius: 20px !important;
            padding: 1.25rem !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255,255,255,0.7) !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.04) !important;
            animation: bubbleFadeIn 0.4s ease-out both;
            max-width: 80% !important;
        }
        
        /* User bubbles */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div[data-testid="stFormatContainer"] {
            background: rgba(255, 255, 255, 0.65) !important;
            margin-right: 0.75rem !important;
            margin-left: 15% !important;
            border-bottom-right-radius: 4px !important;
        }
        
        /* Assistant bubbles */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div[data-testid="stFormatContainer"] {
            background: rgba(255, 255, 255, 0.45) !important;
            margin-left: 0.75rem !important;
            margin-right: 15% !important;
            border-bottom-left-radius: 4px !important;
        }
        
        @keyframes bubbleFadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ── MODE BADGE (Top Right) ── */
        .mode-badge {
            position: fixed;
            top: 1.25rem;
            right: 1.5rem;
            z-index: 1001;
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(14px) saturate(2);
            -webkit-backdrop-filter: blur(14px) saturate(2);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 999px;
            padding: 10px 20px;
            font-size: 0.85rem;
            font-weight: 700;
            color: #1e293b;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            display: flex;
            align-items: center;
            animation: slideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        /* ── CUSTOM SCROLLBAR ── */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.2); }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_chapters(db):
    """Returns the list of chapter names stored in the vector DB for the active book."""
    import chromadb.errors
    try:
        where_filter = None
        if "selected_book" in st.session_state and st.session_state.selected_book:
            where_filter = {"source": st.session_state.selected_book}
            
        sample_data = db.get(limit=1000, include=["metadatas"], where=where_filter)
        raw_chapters = list(set(meta["chapter"] for meta in sample_data["metadatas"] if meta and "chapter" in meta))
        
        def chapter_sort_key(ch):
            if ch.lower() == "inleiding": return 0
            match = re.search(r'\d+', ch)
            return int(match.group()) if match else 999
            
        return ["All Chapters"] + sorted(raw_chapters, key=chapter_sort_key)
    except chromadb.errors.NotFoundError:
        return ["Database Empty - Please wait for build_vector_db.py to finish"]
    except Exception:
        return ["All Chapters"]

def retrieve_documents(prompt, selected_chapter, chapters, db):
    """Retrieves relevant document chunks from ChromaDB using chapter filtering or semantic search."""
    import chromadb.errors
    try:
        where_filter = {}
        if "selected_book" in st.session_state and st.session_state.selected_book:
            where_filter["source"] = st.session_state.selected_book

        target_chapter = None
        if selected_chapter != "All Chapters":
            target_chapter = selected_chapter
        else:
            # Auto-detect chapter from prompt string
            match = re.search(r'(?i)(chapter|hoofdstuk)\s*(\d+)', prompt)
            if match:
                inferred = f"Hoofdstuk {match.group(2)}"
                if inferred in chapters:
                    target_chapter = inferred

        if target_chapter:
            # Fetch entire chapter
            where_filter["chapter"] = target_chapter
            chapter_data = db.get(where=where_filter)
            docs = [Document(page_content=d, metadata=m) for d, m in zip(chapter_data["documents"], chapter_data["metadatas"])] if chapter_data and chapter_data["documents"] else []
            docs.sort(key=lambda x: x.metadata.get("page", 0))
            return docs
        else:
            # Focused semantic search
            search_kwargs = {"k": 5}
            if where_filter:
                search_kwargs["filter"] = where_filter
            retriever = db.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
            return retriever.invoke(prompt)
    except chromadb.errors.NotFoundError:
        return []
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def build_prompt(prompt, docs, summary_level, response_style):
    """Builds the final prompt string from retrieved document chunks and user settings."""
    context_text = "\n\n".join([f"--- Chapter: {doc.metadata.get('chapter', 'Unknown')} | Page: {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}" for doc in docs])
    
    detail_instruction = "Provide a comprehensive, detailed breakdown covering all aspects of the answer."
    if summary_level == "Low":
        detail_instruction = "Provide a concise, high-level summary of the most important points for the overall answer. Keep it brief."
    detail_instruction += " This level of detail applies to BOTH the textbook summary AND the General Medical Knowledge section."

    tone_instruction = "Ensure the tone and language complexity is academic, professional, and highly detailed."
    if response_style == "Simple":
        tone_instruction = "Rewrite all information using simple, everyday, colloquial language. Avoid dense medical jargon where possible, use analogies if helpful, and make it very easy to understand for a layperson. Do not lose factual accuracy."
    tone_instruction += " This complexity of language applies to BOTH the textbook summary AND the General Medical Knowledge section."

    system_template = """
    You are an expert Anatomy and Physiology professor and a highly skilled editor.
    Your task is to answer the user's question based on the Dutch textbook excerpt provided.
    
    CRITICAL INSTRUCTION:
    First, you MUST always summarize whatever information IS present in the provided text excerpt that relates to the user's query, even if it does not fully answer their question. 
    Write this section clearly based strictly on the excerpt.
    
    Then, if the excerpt did NOT fully answer the user's question or lacked the core information, you MUST create a new paragraph starting with "**General Medical Knowledge:**". In this section, provide the full correct answer to the user's query using your own broad medical knowledge.
    
    Please provide all answers in English.
    
    LEVEL OF DETAIL INSTRUCTION: {detail_instruction}
    
    COMPLEXITY OF LANGUAGE INSTRUCTION: {tone_instruction}
    
    TEXT EXCERPT:
    {context}
    
    QUESTION: {question}
    """
    return PromptTemplate.from_template(system_template).format(
        detail_instruction=detail_instruction,
        tone_instruction=tone_instruction,
        context=context_text, 
        question=prompt
    )

def execute_llm_stream(llm_chain, final_prompt, message_placeholder, docs):
    """Streams the LLM response, falling back through the model chain on quota errors."""
    try:
        response = ""
        last_err = None
        for fallback_llm in llm_chain:
            try:
                stream_gen = fallback_llm.stream(final_prompt)
                
                # Deconstruct stream elements to pure text
                def generate_chunks(gen):
                    for chunk in gen:
                        if isinstance(chunk.content, list):
                            for block in chunk.content:
                                if isinstance(block, dict) and "text" in block:
                                    yield block["text"]
                                elif isinstance(block, str):
                                    yield block
                        else:
                            yield str(chunk.content)
                            
                chunk_iterator = generate_chunks(stream_gen)
                
                # Instantly catch API 429 errors
                first_chunk = next(chunk_iterator)
                
                def stream_with_first_chunk(first, rest_iter):
                    if len(first) > 50:
                        for piece in first.split(" "):
                            yield piece + " "
                            time.sleep(0.015)
                    else:
                        yield first
                        
                    for x in rest_iter:
                        if len(x) > 50:
                            for piece in x.split(" "):
                                yield piece + " "
                                time.sleep(0.015)
                        else:
                            yield x
                    
                full_stream = stream_with_first_chunk(first_chunk, chunk_iterator)
                response = message_placeholder.write_stream(full_stream)
                break # Success!
                
            except StopIteration:
                last_err = Exception("Empty response from model")
                continue
            except Exception as e:
                last_err = e
                # Fallback trigger condition
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    response = "" # Wipe failure and try next model
                    continue
                else:
                    raise e
                    
        if not response and last_err:
            raise last_err

        # Append source citations at the end
        sources = set([f"Page {doc.metadata.get('page')}" for doc in docs])
        response += f"\n\n**(Sources: {', '.join(sorted(sources))})**"
        message_placeholder.markdown(response)
        return response
        
    except Exception as e:
        print("\n=== LLM API ERROR ===")
        traceback.print_exc()
        print("=====================\n")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            response = "⚠️ **API Quota Exceeded:** You are using the Free Tier of the Google AI API. Please wait ~45 seconds and try again."
        else:
            response = f"⚠️ **An error occurred:** {str(e)}"
        
        message_placeholder.markdown(response)
        return response

def auto_scroll():
    """Injects JavaScript to scroll to the latest chat message."""
    js = """
    <script>
        function scroll() {
            var elements = window.parent.document.querySelectorAll('.stChatMessage');
            if (elements.length > 0) {
                elements[elements.length - 1].scrollIntoView({behavior: 'smooth'});
            } else {
                window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
            }
        }
        setTimeout(scroll, 100);
    </script>
    """
    st.components.v1.html(js, height=0)

def render_sidebar_library():
    """Renders the book library and PDF uploader inside the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

            /* Force the gradient on every Streamlit container */
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"],
            .main .block-container { background: transparent !important; }

            html {
                background: linear-gradient(135deg,
                    #c9b8f0 0%,
                    #f7c5b0 45%,
                    #aee4c8 100%
                ) fixed !important;
                min-height: 100%;
            }

            /* Sidebar: frosted glass over the gradient */
            /* Sidebar: frosted glass */
            /* Top-level container gets the frosted background */
            section[data-testid="stSidebar"] {
                background: rgba(255,255,255,0.30) !important;
                backdrop-filter: blur(28px) saturate(2.0) !important;
                -webkit-backdrop-filter: blur(28px) saturate(2.0) !important;
                border-right: 1px solid rgba(255,255,255,0.5) !important;
            }
            /* All inner wrapper divs must be transparent so gradient shows through */
            section[data-testid="stSidebar"] > div,
            section[data-testid="stSidebar"] > div > div,
            section[data-testid="stSidebar"] > div > div > div,
            section[data-testid="stSidebar"] > div > div > div > div,
            [data-testid="stSidebarContent"],
            [data-testid="stSidebarUserContent"],
            [data-testid="stSidebar"] .stVerticalBlock,
            [data-testid="stSidebar"] .element-container {
                background: transparent !important;
            }

            .sidebar-title {
                font-family: 'Inter', sans-serif;
                font-size: 1.5rem;
                font-weight: 900;
                color: #0f172a;
                margin-bottom: 1.5rem;
                letter-spacing: -0.02em;
            }
            .lib-label {
                font-family: 'Inter', sans-serif;
                font-size: 0.72rem; font-weight: 700;
                letter-spacing: 0.1em; text-transform: uppercase;
                color: #64748b; margin-bottom: 0.8rem;
                display: flex; align-items: center;
            }
            .lib-label span {
                text-transform: none !important;
                font-size: 0.65rem !important;
                font-weight: 500 !important;
                letter-spacing: normal !important;
                margin-left: 5px;
                opacity: 0.8;
            }
            .sidebar-caption {
                font-family: 'Inter', sans-serif;
                font-size: 0.85rem;
                color: #64748b;
                margin-bottom: 1rem;
            }
            .lib-label::before {
                content: '📚';
                margin-right: 6px;
                font-size: 0.8rem;
            }
            .book-item {
                font-family: 'Inter', sans-serif;
                font-size: 0.85rem; font-weight: 600;
                color: #64748b; margin-bottom: 4px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                display: flex; align-items: center;
            }
            .dot-a {
                display:inline-block; width:7px; height:7px;
                border-radius:50%; background:#22c55e;
                margin-right:7px; vertical-align:middle;
                box-shadow:0 0 5px #22c55e80;
            }
            .dot-i {
                display:inline-block; width:7px; height:7px;
                border-radius:50%; background:#cbd5e1;
                margin-right:7px; vertical-align:middle;
            }
            section[data-testid="stSidebar"] .stButton > button {
                font-family: 'Inter', sans-serif !important;
                font-size: 0.75rem !important; font-weight: 600 !important;
                border-radius: 8px !important; padding: 0.25rem 0.45rem !important;
                background: rgba(255,255,255,0.5) !important;
                border: 1px solid rgba(255,255,255,0.65) !important;
                color: #64748b !important;
            }
            section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: rgba(99,102,241,0.85) !important;
                color: white !important;
                border: none !important;
            }
            /* Bottom padding so library items never slip behind the pinned upload bar */
            [data-testid="stSidebarUserContent"] {
                padding-bottom: 130px !important;
            }
            /* Override dropzone instruction text using pseudo-element */
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
                display: none !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
                content: '📤 Drop a PDF Book';
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem;
                color: #64748b;
                font-weight: 500;
            }

            /* ── Transparent drag-and-drop uploader ── */
            section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: transparent !important;
                border: none !important;
                padding: 0 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: rgba(255,255,255,0.25) !important;
                border: 1.5px dashed rgba(148,163,184,0.5) !important;
                border-radius: 12px !important;
                backdrop-filter: blur(8px) !important;
                -webkit-backdrop-filter: blur(8px) !important;
                box-shadow: none !important;
                padding: 12px !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneIcon"] {
                display: none !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
                background: rgba(255,255,255,0.40) !important;
                border-color: #3b82f6 !important;
            }
            /* Hide the "Limit 200MB per file • PDF" helper text */
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
            section[data-testid="stSidebar"] .stFileUploader label {
                display: none !important;
            }
            /* Style the main "Drag and drop file here" text */
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {
                font-family: 'Inter', sans-serif !important;
                font-size: 0.78rem !important;
                color: #64748b !important;
                font-weight: 500 !important;
            }
            /* Make Browse files button an icon */
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                font-size: 0 !important;
                padding: 0 !important;
                width: 40px !important;
                height: 48px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: none !important;
                margin: 0 auto !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button::before {
                content: '📂';
                font-size: 2rem !important;
                color: #3b82f6 !important;
                visibility: visible !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover::before {
                transform: scale(1.1);
                transition: transform 0.2s ease;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # ── Check upload state FIRST — takes over entire sidebar when active ──
        upload_state = st.session_state.get("upload_state", "idle")

        if upload_state in ("review", "processing"):
            # ── CHAPTER REVIEW MODE: takes over the full sidebar ──────────────
            fname = st.session_state.get("upload_filename", "book.pdf")

            if upload_state == "review":
                total_pages = st.session_state.get("upload_total_pages", "?")
                draft = st.session_state.get("chapter_draft", [])
                book_path = st.session_state.get("upload_book_path", "")

                hdr_col, btn_col = st.columns([3, 1.5])
                with hdr_col:
                    st.markdown(f'<p class="lib-label">📋 Chapter Editor</p>', unsafe_allow_html=True)
                with btn_col:
                    if book_path and st.button("📂 Open PDF", key="open_review_pdf", use_container_width=True):
                        os.system(f"open '{book_path}'")

                st.caption(f"**{fname}** — **{len(draft)}** chapters | **{total_pages}** pages total")
                st.markdown("")

                # Column header
                h1, h2, h3, h4 = st.columns([3, 1.3, 1.3, 0.6])
                h1.markdown("<small style='color:#64748b'>Name</small>", unsafe_allow_html=True)
                h2.markdown("<small style='color:#64748b'>From</small>", unsafe_allow_html=True)
                h3.markdown("<small style='color:#64748b'>To</small>", unsafe_allow_html=True)

                new_draft = []
                for i, ch in enumerate(draft):
                    c1, c2, c3, c4 = st.columns([3, 1.3, 1.3, 0.6])
                    
                    # Ensure values don't exceed max_pages (avoids StreamlitValueAboveMaxError)
                    m_p = int(total_pages) if total_pages != "?" else 9999
                    curr_start = min(max(1, ch.get("start_page", 1)), m_p)
                    curr_end = min(max(1, ch.get("end_page", m_p)), m_p)
                    ch_id = ch.get("id", f"idx_{i}") # Fallback to index if no ID found

                    with c1:
                        new_name = st.text_input(
                            f"Name {i}", value=ch.get("name", ""), label_visibility="collapsed",
                            key=f"ch_name_{ch_id}", placeholder="Chapter name"
                        )
                    with c2:
                        new_start = st.number_input(
                            f"Start {i}", value=curr_start, min_value=1,
                            max_value=m_p,
                            label_visibility="collapsed", key=f"ch_start_{ch_id}", step=1
                        )
                    with c3:
                        new_end = st.number_input(
                            f"End {i}", value=curr_end, min_value=1,
                            max_value=m_p,
                            label_visibility="collapsed", key=f"ch_end_{ch_id}", step=1
                        )
                    with c4:
                        if st.button("🗑️", key=f"ch_del_{ch_id}", use_container_width=True):
                            draft.pop(i)
                            st.session_state.chapter_draft = draft
                            st.rerun()
                    new_draft.append({"id": ch_id, "name": new_name, "start_page": int(new_start), "end_page": int(new_end)})

                # Sort and recompute cascade end pages from start pages
                new_draft = sorted(new_draft, key=lambda x: x["start_page"])
                for i in range(len(new_draft) - 1):
                    # Only auto-fix end page if it doesn't match user intent (start of next chapter - 1)
                    if new_draft[i]["end_page"] >= new_draft[i + 1]["start_page"]:
                        new_draft[i]["end_page"] = new_draft[i + 1]["start_page"] - 1
                st.session_state.chapter_draft = new_draft

                if st.button("＋ Add Chapter", use_container_width=True):
                    import uuid
                    m_p = int(total_pages) if total_pages != "?" else 9999
                    if new_draft:
                        last = new_draft[-1]
                        new_s = min(last["end_page"] + 1, m_p)
                        new_e = m_p
                    else:
                        new_s = 1
                        new_e = m_p
                    
                    st.session_state.chapter_draft.append({
                        "id": str(uuid.uuid4())[:8],
                        "name": f"Chapter {len(new_draft)+1}", 
                        "start_page": new_s, 
                        "end_page": new_e
                    })
                    st.rerun()

                st.markdown("")
                col_x, col_ok = st.columns(2)
                with col_x:
                    if st.button("✗ Cancel", use_container_width=True):
                        # Cleanup the PDF we saved at scan time since user is cancelling
                        delete_book_data(fname)
                        for k in ["upload_state", "upload_filename", "upload_file_bytes", "chapter_draft", "upload_total_pages", "upload_book_path"]:
                            st.session_state.pop(k, None)
                        st.rerun()
                with col_ok:
                    if st.button("✅ Embed", type="primary", use_container_width=True):
                        st.session_state.upload_state = "processing"
                        st.rerun()

            elif upload_state == "processing":
                st.markdown('<p class="lib-label">🔄 Embedding</p>', unsafe_allow_html=True)
                st.caption(f"Processing **{fname}**... please wait.")
                draft = st.session_state.get("chapter_draft", [])
                chapter_map = {ch["start_page"]: ch["name"] for ch in draft}

                import sys, io as _io
                if "scripts" not in sys.path:
                    sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
                from scripts.build_vector_db import build_vector_db as build_db_func

                file_bytes = st.session_state.get("upload_file_bytes", b"")
                if not os.path.exists("books"):
                    os.makedirs("books")
                with open(os.path.join("books", fname), "wb") as f:
                    f.write(file_bytes)

                bar = st.progress(0.0)
                msg = st.empty()
                def update_progress(pct, text):
                    msg.caption(text)
                    bar.progress(min(pct, 1.0))

                try:
                    st.cache_resource.clear()
                    build_db_func(_io.BytesIO(file_bytes), progress_callback=update_progress,
                                  source=fname, chapter_map=chapter_map)
                    st.balloons()
                    for k in ["upload_state", "upload_filename", "upload_file_bytes", "chapter_draft"]:
                        st.session_state.pop(k, None)
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    msg.error(f"❌ Error: {e}")
                    # Cleanup remnants on failure (files, partial embeddings, etc.)
                    delete_book_data(fname)
                    st.session_state.upload_state = "idle"

            return  # Don't render the library or upload area below


        if not os.path.exists("books"):
            os.makedirs("books")

        books = [f for f in os.listdir("books") if f.endswith(".pdf")]
        is_proc = st.session_state.get("is_processing", False)

        # ── NORMAL LIBRARY VIEW ────────────────────────────────────────────
        # Sidebar Title
        st.markdown('<div class="sidebar-title">PROFOOT</div>', unsafe_allow_html=True)
        
        lib_text = "Library"
        if not books:
            lib_text += ' <span>(No books yet — drop one below ↓)</span>'
        st.markdown(f'<p class="lib-label">{lib_text}</p>', unsafe_allow_html=True)

        if books:
            for book in books:
                is_active = st.session_state.get("selected_book") == book
                
                # Three-column layout for Selection, Open, Delete
                c1, c2, c3 = st.columns([3, 0.6, 0.6])
                
                with c1:
                    # Clicking the filename selects the book
                    lbl = f"✅ {book}" if is_active else book
                    if st.button(lbl, key=f"sel_{book}", use_container_width=True,
                                 type="primary" if is_active else "secondary", disabled=is_proc):
                        new_sel = None if is_active else book
                        st.session_state.selected_book = new_sel
                        save_selection(new_sel or "")
                        st.rerun()
                
                with c2:
                    if st.button("📂", key=f"open_{book}", use_container_width=True, disabled=is_proc):
                        book_path = os.path.abspath(os.path.join("books", book))
                        os.system(f"open '{book_path}'")
                
                with c3:
                    if st.button("🗑️", key=f"del_{book}", use_container_width=True, disabled=is_proc):
                        delete_book_data(book)
                        if st.session_state.get("selected_book") == book:
                            st.session_state.selected_book = None
                            save_selection("")
                        st.rerun()
                st.markdown('<div style="margin-top: -10px;"></div>', unsafe_allow_html=True)

        # Render the uploader (will be moved to the bottom by JS below)
        st.markdown('<div class="upload-anchor">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop a PDF Book", type=["pdf"], key="sidebar_uploader", label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # JS: inject via st.markdown so it runs in the MAIN page context (not a sandboxed iframe).
        # It physically moves .upload-anchor to be a direct child of the sidebar <section>,
        # then pins it with position:absolute;bottom:0 — bypassing the overflow:auto container.
        st.markdown("""
        <script>
        (function pinUploader() {
            function doPin() {
                var sidebar = document.querySelector('section[data-testid="stSidebar"]');
                var anchor  = document.querySelector('section[data-testid="stSidebar"] .upload-anchor');
                if (!sidebar || !anchor) return;

                // Walk up until the direct child of <section>
                var el = anchor;
                while (el.parentElement && el.parentElement !== sidebar) {
                    el = el.parentElement;
                }
                if (el === sidebar) {
                    // Already moved — just ensure styles are applied
                    el.style.cssText = 'position:absolute!important;bottom:0!important;left:0!important;right:0!important;width:100%!important;padding:10px 1rem 12px!important;background:rgba(255,255,255,0.22)!important;backdrop-filter:blur(18px) saturate(1.8)!important;border-top:1px solid rgba(255,255,255,0.50)!important;z-index:100!important;box-sizing:border-box!important;';
                    return;
                }

                sidebar.style.position = 'relative';
                el.style.cssText = 'position:absolute!important;bottom:0!important;left:0!important;right:0!important;width:100%!important;padding:10px 1rem 12px!important;background:rgba(255,255,255,0.22)!important;backdrop-filter:blur(18px) saturate(1.8)!important;border-top:1px solid rgba(255,255,255,0.50)!important;z-index:100!important;box-sizing:border-box!important;';
                sidebar.appendChild(el);
            }
            [100, 400, 900, 2500].forEach(function(d){ setTimeout(doPin, d); });
        })();
        </script>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            if st.button("⚡ Scan Chapters", type="primary", use_container_width=True):
                import sys
                if "scripts" not in sys.path:
                    sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
                from scripts.build_vector_db import detect_chapters
                with st.spinner("Scanning... (~30–60 sec for image-based PDFs)"):
                    detected, total_pages = detect_chapters(uploaded_file)

                # Save to disk now so the Open button works during review
                if not os.path.exists("books"):
                    os.makedirs("books")
                file_bytes = uploaded_file.getvalue()
                book_path = os.path.join("books", uploaded_file.name)
                with open(book_path, "wb") as f:
                    f.write(file_bytes)

                # Compute end_page for each chapter (next chapter start - 1, last = total pages)
                import uuid
                for i, ch in enumerate(detected):
                    ch["id"] = str(uuid.uuid4())[:8]  # Unique ID for widget keys
                    if i + 1 < len(detected):
                        ch["end_page"] = detected[i + 1]["start_page"] - 1
                    else:
                        ch["end_page"] = total_pages

                st.session_state.upload_state = "review"
                st.session_state.upload_filename = uploaded_file.name
                st.session_state.upload_file_bytes = file_bytes
                st.session_state.upload_book_path = os.path.abspath(book_path)
                st.session_state.upload_total_pages = total_pages
                st.session_state.chapter_draft = detected
                st.rerun()

def render_landing_page():
    # Load persistent selection
    if "selected_book" not in st.session_state:
        st.session_state.selected_book = load_selection()
    selected_book = st.session_state.get("selected_book")

    # Landing-specific glassmorphic CSS
    st.markdown(
        """
        <style>
        /* ─── HERO TRANSITIONS ─── */
        .hero-container {
            transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .hero-container.idle {
            padding-top: 12vh;
        }
        .hero-container.active {
            padding-top: 2vh;
        }

        .hero-title {
            font-family: 'Inter', sans-serif !important;
            text-align: center !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            letter-spacing: -0.04em !important;
            line-height: 1.1 !important;
            margin: 0 auto !important;
            transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        .hero-container.idle .hero-title {
            font-size: 6.5rem !important;
        }
        .hero-container.active .hero-title {
            font-size: 4.2rem !important;
        }

        .hero-sub {
            font-family: 'Inter', sans-serif;
            text-align: center;
            font-size: 1.25rem;
            color: #64748b;
            font-weight: 500;
            margin-top: 12px;
            margin-bottom: 2.5rem;
            transition: opacity 0.5s ease;
        }
        .hero-container.idle .hero-sub { opacity: 1; }
        .hero-container.active .hero-sub { opacity: 0.8; margin-bottom: 1.5rem; }

        /* ─── STATUS BADGE ─── */
        .status-badge {
            display: flex;
            align-items: center;
            width: fit-content;
            margin: 0 auto 3rem auto;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            border-radius: 999px;
            padding: 6px 18px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .hero-container.active .status-badge {
            margin-bottom: 4rem;
        }
        .status-badge .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-badge.active { 
            background: rgba(34, 197, 94, 0.15);
            color: #15803d;
            border: 1px solid rgba(34, 197, 94, 0.25);
        }
        .status-badge.active .dot { background: #22c55e; }
        .status-badge.idle {
            background: rgba(239, 68, 68, 0.15);
            color: #dc2626;
            border: 1px solid rgba(239, 68, 68, 0.25);
        }
        .status-badge.idle .dot { background: #ef4444; }


        /* ─── GLASS CARDS ─── */
        .glass-card {
            background: rgba(255,255,255,0.55);
            backdrop-filter: blur(24px) saturate(1.8);
            -webkit-backdrop-filter: blur(24px) saturate(1.8);
            border: 1px solid rgba(255,255,255,0.8);
            border-radius: 28px;
            padding: 48px 24px 24px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.06);
            margin-bottom: 16px;
            animation: cardAppear 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .card-l { animation-delay: 0.1s; }
        .card-r { animation-delay: 0.25s; }

        @keyframes cardAppear {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card-emoji { font-size: 3.5rem; display: block; margin-bottom: 20px; }
        .card-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.2rem; font-weight: 800;
            color: #1e293b; margin-bottom: 12px;
            letter-spacing: -0.02em;
        }
        .card-desc {
            font-family: 'Inter', sans-serif;
            font-size: 1rem; color: #64748b;
            line-height: 1.6; margin-bottom: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    hero_class = "active" if selected_book else "idle"
    st.markdown(f"""
        <div class="hero-container {hero_class}">
            <h1 class="hero-title">Profoot - AI</h1>
            <p class="hero-sub">Pedicure assistant</p>
    """, unsafe_allow_html=True)

    # Status badge
    if selected_book:
        st.markdown(
            f'<div class="status-badge active"><span class="dot"></span>{selected_book} is active</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge idle"><span class="dot"></span>Select a book in the sidebar to begin</div>',
            unsafe_allow_html=True,
        )
    
    st.markdown('</div>', unsafe_allow_html=True) # End hero-container

    # Mode cards - only visible when a book is selected
    if selected_book:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown(
                """<div class="glass-card card-l">
                    <span class="card-emoji">📖</span>
                    <p class="card-title">Study</p>
                    <p class="card-desc">Ask questions, get detailed answers,<br>and explore your textbook chapter by chapter.</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "Launch Study →", key="launch_study",
                use_container_width=True, type="primary"
            ):
                st.session_state.app_mode = "study"
                st.rerun()

        with col_r:
            st.markdown(
                """<div class="glass-card card-r">
                    <span class="card-emoji">📝</span>
                    <p class="card-title">Test</p>
                    <p class="card-desc">Generate a personalised quiz,<br>set a timer, and check your knowledge.</p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "Launch Test →", key="launch_test",
                use_container_width=True, type="primary"
            ):
                st.session_state.app_mode = "test"
                st.rerun()


def run_test_mode():
    if "test_phase" not in st.session_state:
        st.session_state.test_phase = "config"
        
    with st.sidebar:
        # We handle Main Menu button globally in main() sidebar
        pass

    # Book Reminder Badge (Top Right)
    sel_book = st.session_state.get("selected_book", "No Book Selected")
    st.markdown(f'<div class="mode-badge">📖 {sel_book}</div>', unsafe_allow_html=True)

    st.title("📝 Test Mode")
    
    if st.session_state.test_phase == "config":
        st.markdown('<div class="glass-card" style="padding: 40px; margin: 2rem auto; max-width: 600px;">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>🎯 Test Configuration</h2>", unsafe_allow_html=True)
        
        db = load_db()
        all_chapters = get_chapters(db)
        chapter_opts = [ch for ch in all_chapters if ch != "All Chapters"]
        
        selected_chapters = st.multiselect("Select Focus Chapters (Optional):", chapter_opts)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            q_count = st.selectbox("Questions:", [5, 10, 15, 20, 25, 30, 40, 50], index=1)
        with col2:
            o_count = st.selectbox("Options:", [3, 4, 5], index=1)
        with col3:
            t_length = st.selectbox("Timer (min):", [10, 15, 20, 30, 45, 60], index=1)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate My Quiz →", use_container_width=True, type="primary"):
            st.session_state.test_config = {
                "chapters": selected_chapters,
                "q_count": q_count,
                "o_count": o_count,
                "t_length": t_length
            }
            st.session_state.test_phase = "loading"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.test_phase == "loading":
        st.info("🩺 Preparing your test...")
        
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        debug_info = st.empty()
        
        def update_progress(current_pct, fun_msg, debug_msg=""):
            status_text.markdown(f"#### {fun_msg}")
            progress_bar.progress(current_pct)
            debug_info.caption(f"⚙️ Developer Info: {debug_msg}")
        db = load_db()
        llms = load_llms()
        
        import test_utils
        import chromadb.errors
        import random
        
        chapter_docs = {}
        total_chunks = 0
        try:
            if st.session_state.test_config["chapters"]:
                for ch in st.session_state.test_config["chapters"]:
                    where_filter = {"chapter": ch}
                    if st.session_state.get("selected_book"):
                        where_filter["source"] = st.session_state.selected_book
                    chapter_data = db.get(where=where_filter, include=["metadatas", "documents"])
                    if chapter_data and chapter_data["documents"]:
                        docs = [Document(page_content=doc_text, metadata=chapter_data["metadatas"][i]) for i, doc_text in enumerate(chapter_data["documents"])]
                        chapter_docs[ch] = docs
                        total_chunks += len(docs)
            else:
                where_filter = None
                if st.session_state.get("selected_book"):
                    where_filter = {"source": st.session_state.selected_book}
                all_data = db.get(include=["metadatas", "documents"], where=where_filter)
                if all_data and all_data["documents"]:
                    # Group all by chapter
                    for i, doc_text in enumerate(all_data["documents"]):
                        meta = all_data["metadatas"][i]
                        ch = meta.get("chapter", "Unknown Chapter")
                        if ch not in chapter_docs:
                            chapter_docs[ch] = []
                        chapter_docs[ch].append(Document(page_content=doc_text, metadata=meta))
                        total_chunks += 1
                        
            # If "All Chapters" is selected and it's too big, we might want to sample chapters or just use them all.
            # For proportional distribution, we need the counts.
            if len(chapter_docs) == 0:
                raise chromadb.errors.NotFoundError()
                
        except chromadb.errors.NotFoundError:
            st.error("Database is empty or still building! Please wait for `build_vector_db.py` to finish or upload a PDF first.")
            if st.button("Back"):
                st.session_state.test_phase = "config"
                st.rerun()
            return
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                 st.error("Database is empty or still building! Please wait for `build_vector_db.py` to finish or upload a PDF first.")
            else:
                 st.error(f"Error reading database: {e}")
                 
            if st.button("Back"):
                st.session_state.test_phase = "config"
                st.rerun()
            return

        update_progress(0.1, "🩻 Scanning the textbook...", "Documents loaded")
        
        # Calculate Quotas
        total_q_requested = st.session_state.test_config["q_count"]
        quotas = {}
        
        if total_chunks > 0:
            # Initial proportional allocation
            remaining_q = total_q_requested
            for ch, docs in chapter_docs.items():
                proportion = len(docs) / total_chunks
                # Give at least 1 question if it's selected, unless we asked for very few overall
                quota = max(1, int(round(proportion * total_q_requested)))
                quotas[ch] = quota
                remaining_q -= quota
                
            # Adjust rounding errors
            # If we over-allocated, reduce from the largest quota
            while remaining_q < 0:
                largest_ch = max(quotas, key=quotas.get)
                if quotas[largest_ch] > 1:
                    quotas[largest_ch] -= 1
                    remaining_q += 1
                else:
                    break
                    
            # If we under-allocated, add to the largest quota
            while remaining_q > 0:
                largest_ch = max(quotas, key=quotas.get)
                quotas[largest_ch] += 1
                remaining_q -= 1
        
        try:
            api_keys = get_all_api_keys()
            quiz_data = test_utils.generate_mock_test(
                api_keys, 
                st.session_state.get("selected_book", "Unknown Book"),
                chapter_docs, 
                quotas,
                st.session_state.test_config["o_count"],
                progress_callback=update_progress
            )

            st.session_state.test_data = quiz_data
            st.session_state.test_start_time = time.time()
            st.session_state.test_answers = {}
            st.session_state.test_phase = "active"
            st.rerun()
        except Exception as e:
            st.error(f"Failed to generate test. Please try again. Error: {e}")
            if st.button("Back"):
                st.session_state.test_phase = "config"
                st.rerun()

    elif st.session_state.test_phase == "active":
        elapsed = time.time() - st.session_state.test_start_time
        remaining_secs = max(0, (st.session_state.test_config["t_length"] * 60) - elapsed)
        
        # Professional Glassmorphic Timer
        timer_color = "#e11d48" if remaining_secs < 60 else "#334155" # Red if less than 1 min
        js_timer = f"""
        <div id="timer_display" style="
            font-family: 'Inter', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: {timer_color};
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.6);
            border-radius: 18px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        ">
           ⏱️ {int(remaining_secs // 60):02d}:{int(remaining_secs % 60):02d}
        </div>
        <script>
            var remaining = {int(remaining_secs)};
            var el = document.getElementById('timer_display');
            var timer = setInterval(function() {{
                remaining--;
                if (remaining <= 0) {{
                    clearInterval(timer);
                    el.style.color = "#e11d48";
                    el.innerHTML = "⏱️ 00:00 - TIME EXPIRED";
                }} else {{
                    var m = Math.floor(remaining / 60);
                    var s = remaining % 60;
                    if (remaining < 60) el.style.color = "#e11d48";
                    el.innerHTML = "⏱️ " + (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
                }}
            }}, 1000);
        </script>
        """
        st.components.v1.html(js_timer, height=100)
        
        if remaining_secs <= 0:
            st.warning("Time's up! Please finalize and submit.")
        
        with st.form("test_form", border=False):
            for i, q in enumerate(st.session_state.test_data):
                st.markdown(f'''
                    <div class="glass-card" style="padding: 24px; margin-bottom: 20px;">
                        <h4 style="margin-top: 0; color: #1e293b; font-size: 1.1rem;">Question {i+1}</h4>
                        <p style="font-weight: 500; font-size: 1.05rem; margin-bottom: 1.5rem;">{q.get('question', 'Unknown Question')}</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                options = q.get('options', [])
                st.radio("Select your answer:", options, key=f"q_{i}", index=None, label_visibility="collapsed")
                st.markdown("<br>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Finish and Result →", type="primary", use_container_width=True)
            
            if submitted:
                for i in range(len(st.session_state.test_data)):
                    st.session_state.test_answers[i] = st.session_state[f"q_{i}"]
                st.session_state.test_phase = "results"
                st.rerun()
                
    elif st.session_state.test_phase == "results":
        score = 0
        total = len(st.session_state.test_data)
        for i, q in enumerate(st.session_state.test_data):
            if st.session_state.test_answers.get(i) == q.get('correct_answer'):
                score += 1
        
        # Summary Score Card
        pct = (score/total)*100
        st.markdown(f'''
            <div class="glass-card" style="padding: 40px; text-align: center; margin-bottom: 3rem;">
                <h3 style="margin-top: 0; color: #64748b; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 0.1em;">Quiz Performance</h3>
                <h1 style="font-size: 5rem !important; margin: 1rem 0 !important; color: #1e293b !important;">{score} <span style="font-size: 2rem; color: #94a3b8;">/ {total}</span></h1>
                <p style="font-size: 1.2rem; font-weight: 600; color: #3b82f6;">{pct:.1f}% Mastered</p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### Detailed Review")
        
        for i, q in enumerate(st.session_state.test_data):
            user_ans = st.session_state.test_answers.get(i)
            correct_ans = q.get('correct_answer')
            is_correct = user_ans == correct_ans
            
            st.markdown(f'''
                <div class="glass-card" style="padding: 24px; margin-bottom: 2rem; border-left: 6px solid {"#22c55e" if is_correct else "#ef4444"};">
                    <h4 style="margin-top: 0; color: #475569;">Question {i+1}</h4>
                    <p style="font-weight: 600; font-size: 1.1rem; margin-bottom: 1.5rem;">{q.get('question', 'Unknown Question')}</p>
                    <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 12px; margin-bottom: 15px;">
                        <p style="margin: 0; font-size: 0.95rem;">
                            <span style="color: {"#15803d" if is_correct else "#b91c1c"}; font-weight: 700;">Your Answer:</span> {user_ans if user_ans else 'No answer'} 
                            { "✅" if is_correct else "❌" }
                        </p>
                        {f'<p style="margin: 5px 0 0 0; font-size: 0.95rem;"><span style="color: #15803d; font-weight: 700;">Correct Answer:</span> {correct_ans} 🎯</p>' if not is_correct else ""}
                    </div>
                    <p style="font-size: 0.95rem; line-height: 1.5; color: #475569;">
                        <b>Explanation:</b> {q.get('correct_explanation', q.get('explanation', 'No explanation provided.'))}
                    </p>
                </div>
            ''', unsafe_allow_html=True)
            
            incorrect_opts = [opt for opt in q.get('options', []) if opt != correct_ans]
            if incorrect_opts:
                incorrect_explanations = q.get('incorrect_explanations', {})
                with st.expander("Explore the wrong options 🔍"):
                    for opt in incorrect_opts:
                        st.markdown(f"**{opt}**")
                        st.write(incorrect_explanations.get(opt, "No explanation available."))
            
        if st.button("Take Another Quiz", type="primary"):
            st.session_state.test_phase = "config"
            st.session_state.test_answers = {}
            st.rerun()

def run_study_mode():
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
        
    with st.sidebar:
        # Chat History header and New Chat handle in global menu?
        # No, keep it in sidebar but tidy it up.
        st.markdown('<p class="lib-label">🕒 History</p>', unsafe_allow_html=True)
        if st.button("＋ New Chat", use_container_width=True, type="secondary"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        for sess in db_utils.get_recent_sessions():
            is_active = sess['id'] == st.session_state.current_session_id
            btn_type = "primary" if is_active else "secondary"
            if st.button(sess['title'], key=f"btn_{sess['id']}", use_container_width=True, type=btn_type):
                st.session_state.current_session_id = sess['id']
                st.session_state.messages = db_utils.get_messages(sess['id'])
                st.rerun()

    if not os.path.exists(CHROMA_PATH):
        st.error("Vector Database not found! Please run `python scripts/build_vector_db.py` first.")
        st.stop()
        
    # Book Reminder Badge (Top Right)
    sel_book = st.session_state.get("selected_book", "No Book Selected")
    st.markdown(f'<div class="mode-badge">📖 {sel_book}</div>', unsafe_allow_html=True)

    st.markdown("<h1 style='font-size: 3rem !important; margin-bottom: 0.5rem !important;'>🧠 Study Mode</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>Master your material with Ai-driven insights.</p>", unsafe_allow_html=True)

    # Initialize environment
    db = load_db()
    llm_chain = load_llms()
    chapters = get_chapters(db)

    # Action Bar Settings (Control Panel)
    st.markdown('<div class="glass-card" style="padding: 1.5rem; margin-bottom: 2.5rem;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        selected_chapter = st.selectbox("🎯 Focus Chapter:", chapters)
    with c2:
        summary_level = st.radio("📊 Depth:", options=["Low", "High"], horizontal=True, index=0)
    with c3:
        response_style = st.radio("🎨 Style:", options=["Standard", "Simple"], horizontal=True, index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Render existing chat
    if "messages" not in st.session_state or not st.session_state.messages:
        # Load from Database first
        st.session_state.messages = db_utils.get_messages(st.session_state.current_session_id)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Wait for user prompt
    if prompt := st.chat_input("What is the function of the heart?"):
        st.chat_message("user").markdown(prompt)
        
        # If this is absolute first message, generate title and init session
        if not st.session_state.messages:
            title = db_utils.generate_chat_title(prompt)
            db_utils.save_session(st.session_state.current_session_id, title)
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        db_utils.save_message(st.session_state.current_session_id, "user", prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ **Searching textbook for knowledge...**")
            auto_scroll()
            
            # Execute Pipeline
            docs = retrieve_documents(prompt, selected_chapter, chapters, db)
            
            if not docs:
                response = "I couldn't find any relevant information in the book for that question."
                message_placeholder.markdown(response)
            else:
                message_placeholder.markdown("🧠 **Reading contexts & thinking...**")
                final_prompt = build_prompt(prompt, docs, summary_level, response_style)
                response = execute_llm_stream(llm_chain, final_prompt, message_placeholder, docs)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        db_utils.save_message(st.session_state.current_session_id, "assistant", response)
        auto_scroll()

# --- Main Application Execution ---
def main():
    st.set_page_config(page_title="Profoot - AI", page_icon="🦶", layout="centered")
    
    # Robust API Key Handling for Local Users
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("GOOGLE_API_KEY", "")
        
    if not api_key:
        st.title("🔑 Initial Setup Required")
        st.warning("No Google API Key detected. Please enter your Gemini API Key below to use this application.")
        st.markdown("You can get a free API key from [Google AI Studio](https://aistudio.google.com/).")
        
        user_key = st.text_input("Gemini API Key:", type="password")
        if st.button("Save & Continue", type="primary"):
            if user_key.startswith("AIza"):
                st.session_state["GOOGLE_API_KEY"] = user_key
                os.environ["GOOGLE_API_KEY"] = user_key
                st.success("API Key saved! Loading application...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid API Key format. It should start with 'AIza'.")
        return # Block app loading until key is provided

    db_utils.init_db()
    
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
        
    apply_global_styles()

    # Global Sidebar
    with st.sidebar:
        if st.session_state.app_mode:
            if st.button("🏠 Main Menu", use_container_width=True):
                # Cleanup specific mode states
                if "test_phase" in st.session_state: del st.session_state.test_phase
                st.session_state.app_mode = None
                st.rerun()
            st.markdown("---")
        else:
            render_sidebar_library()
        
    if st.session_state.app_mode == "study":
        run_study_mode()
    elif st.session_state.app_mode == "test":
        run_test_mode()
    else:
        render_landing_page()

if __name__ == "__main__":
    main()
