import sqlite3, json, os
from datetime import datetime

DB_PATH = "data/hr_app.db"

def _conn():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = _conn()
    c.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, upload_time TEXT, chunk_count INTEGER, file_hash TEXT UNIQUE)")
    c.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT, sources TEXT, timestamp TEXT)")
    c.commit(); c.close()

def add_document(filename, chunk_count, file_hash):
    c = _conn()
    c.execute("INSERT OR REPLACE INTO documents (filename, upload_time, chunk_count, file_hash) VALUES (?, ?, ?, ?)",
              (filename, datetime.now().isoformat(), chunk_count, file_hash))
    c.commit(); c.close()

def get_all_documents():
    c = _conn()
    rows = c.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]

def delete_document(doc_id):
    c = _conn()
    row = c.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row:
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        c.commit()
    c.close()
    return row["filename"] if row else None

def document_exists(file_hash):
    c = _conn()
    row = c.execute("SELECT 1 FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
    c.close()
    return row is not None

def add_chat_entry(question, answer, sources):
    c = _conn()
    c.execute("INSERT INTO chat_history (question, answer, sources, timestamp) VALUES (?, ?, ?, ?)",
              (question, answer, json.dumps(sources), datetime.now().isoformat()))
    c.commit(); c.close()

def get_chat_history():
    c = _conn()
    rows = c.execute("SELECT * FROM chat_history ORDER BY timestamp ASC").fetchall()
    c.close()
    return [{**dict(r), "sources": json.loads(r["sources"])} for r in rows]

def clear_chat_history():
    c = _conn()
    c.execute("DELETE FROM chat_history")
    c.commit(); c.close()