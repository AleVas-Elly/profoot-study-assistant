import sqlite3
import os
import uuid
import datetime
from langchain_core.prompts import PromptTemplate

DB_PATH = "chat_history.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Creates the SQLite tables for sessions, messages, and generated test questions."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                updated_at TIMESTAMP
            )
        """)
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)
        # Past questions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS past_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                chapter TEXT,
                question_text TEXT,
                timestamp TIMESTAMP
            )
        """)
        
        # Migration: Add source column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE past_questions ADD COLUMN source TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists

        conn.commit()


def save_session(session_id, title=None):
    """Saves or updates a session, keeping only the 10 most recent."""
    now = datetime.datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if cursor.fetchone():
            if title:
                cursor.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, session_id))
            else:
                cursor.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        else:
            final_title = title or "New Conversation"
            cursor.execute("INSERT INTO sessions (id, title, updated_at) VALUES (?, ?, ?)", (session_id, final_title, now))

        # Keep only the 10 most recent sessions
        cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC LIMIT -1 OFFSET 10")
        old_sessions = cursor.fetchall()
        for old in old_sessions:
            cursor.execute("DELETE FROM sessions WHERE id = ?", (old[0],))
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (old[0],)) # Manually enforce cascade
            
        conn.commit()

def save_message(session_id, role, content):
    """Saves a message, keeping only the 20 most recent per session."""
    now = datetime.datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        # Keep only the 20 most recent messages per session
        cursor.execute("SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT -1 OFFSET 20", (session_id,))
        old_messages = cursor.fetchall()
        for old in old_messages:
            cursor.execute("DELETE FROM messages WHERE id = ?", (old[0],))
            
        conn.commit()
        
    # Bump the session updated_at so it stays at the top of the history list
    save_session(session_id)

def get_recent_sessions(limit=10):
    """Returns the most recent session IDs and titles."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]

def get_messages(session_id):
    """Returns the message history for a specific session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

def generate_chat_title(first_prompt):
    """Derives a session title from the user's first message."""
    title = first_prompt.strip()
    if title:
        title = title[0].upper() + title[1:]
    if len(title) > 35:
        return title[:32] + "..."
    return title

def save_past_questions(source, chapter, questions):
    """Persists a list of generated questions for deduplication in future quiz runs."""
    now = datetime.datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        for q in questions:
            question_text = q.get('question', '')
            if question_text:
                cursor.execute(
                    "INSERT INTO past_questions (source, chapter, question_text, timestamp) VALUES (?, ?, ?, ?)",
                    (source, chapter, question_text, now)
                )
        conn.commit()

def get_past_questions(source, chapter, limit=20):
    """Fetches recent past questions for a book and chapter to avoid duplicates."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question_text FROM past_questions WHERE source = ? AND chapter = ? ORDER BY id DESC LIMIT ?",
            (source, chapter, limit)
        )
        return [row[0] for row in cursor.fetchall()]

def delete_past_questions_by_source(source):
    """Deletes all stored questions associated with a specific book."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM past_questions WHERE source = ?", (source,))
        conn.commit()


