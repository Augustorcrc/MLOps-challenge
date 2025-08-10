import os
import time
import sqlite3
import re
from typing import Optional, Tuple

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # carpeta raíz del proyecto (ajustá si tu estructura es distinta)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)        # crea ./data si no existe

DB_PATH = str(DATA_DIR / "bible.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS verses (
    ref TEXT PRIMARY KEY,          -- clave normalizada (lower/espacios colapsados)
    ref_raw TEXT NOT NULL,         -- como lo envió el usuario (última vez)
    data_json TEXT NOT NULL,       -- JSON del verso (tal cual devuelto por bible-api)
    request_count INTEGER NOT NULL DEFAULT 0
);
"""


def get_conn():
    """Abre conexión SQLite con timeout y PRAGMAs para mejorar concurrencia."""
    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
            # Espera hasta 5s si hay locks en curso
            conn.execute("PRAGMA busy_timeout=5000;")
            # WAL mejora concurrencia de lecturas simultáneas
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            return conn
        except sqlite3.OperationalError as exc:
            last_err = exc
            if "locked" in str(exc).lower():
                time.sleep(0.2 * (attempt + 1))
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to open SQLite connection")


def init_db():
    """Inicializa el esquema una vez usando un file lock simple entre procesos."""
    lock_path = f"{DB_PATH}.init.lock"
    start = time.time()
    acquired = False
    while not acquired and (time.time() - start) < 10.0:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            acquired = True
        except FileExistsError:
            time.sleep(0.2)
    try:
        with get_conn() as conn:
            conn.executescript(SCHEMA_SQL)
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def norm_ref(s: str) -> str:
    """Normalize a Bible reference string to a canonical cache key.

    Rules applied:
    - Lowercase everything
    - Trim leading/trailing whitespace
    - Collapse internal whitespace
    - Remove whitespace around colon (:) separators
    - Ensure a space between book name and chapter if missing (e.g., "john3:16" -> "john 3:16")
    - Ensure a space between leading number and book name (e.g., "1john" -> "1 john")
    """
    s = s.strip().lower()
    # Normalize whitespace around colon
    s = re.sub(r"\s*:\s*", ":", s)
    # Ensure a space between leading number and book name (e.g., 1john -> 1 john)
    s = re.sub(r"^(\d)\s*([a-z])", r"\1 \2", s)
    # Ensure a space between book name and chapter if missing (e.g., john3:16 -> john 3:16)
    s = re.sub(r"([a-z])\s*(\d)", r"\1 \2", s)
    # Collapse multiple spaces
    s = " ".join(s.split())
    # Map common book abbreviations to canonical names (prefix number preserved)
    aliases = {
        # Gospels + common
        "mt": "matthew", "matt": "matthew",
        "mk": "mark", "mrk": "mark",
        "lk": "luke", "luk": "luke",
        "jn": "john", "jhn": "john", "joh": "john",
        # A few more frequent examples
        "rom": "romans", "ro": "romans",
        "ps": "psalm", "psa": "psalm", "psalms": "psalm",
        "prov": "proverbs", "pr": "proverbs",
        "rev": "revelation", "re": "revelation",
    }
    # Split reference into head (book + chapter) and tail (":verse" if present)
    head, tail = (s.split(":", 1) + [""])[:2]
    tail = (":" + tail) if tail else ""
    parts = head.split()
    if not parts:
        return s
    # Expect last part is chapter number
    if parts[-1].isdigit():
        chapter = parts[-1]
        book_tokens = parts[:-1]
        if not book_tokens:
            return s
        # Handle optional numeric prefix (1/2/3)
        prefix = ""
        base = book_tokens[0]
        if len(book_tokens) >= 2 and book_tokens[0] in {"1", "2", "3"}:
            prefix = book_tokens[0] + " "
            base = book_tokens[1]
        canonical = aliases.get(base, base)
        head_norm = f"{prefix}{canonical} {chapter}"
        return f"{head_norm}{tail}"
    return s


def get_verse(ref_norm: str) -> Optional[Tuple[str, str, int]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT ref_raw, data_json, request_count FROM verses WHERE ref = ?", (ref_norm,))
        row = cur.fetchone()
        if not row:
            return None
        return row[0], row[1], row[2]


def insert_or_update(ref_norm: str, ref_raw: str, data_json: str, is_new: bool):
    with get_conn() as conn:
        if is_new:
            conn.execute(
                "INSERT INTO verses (ref, ref_raw, data_json, request_count) VALUES (?, ?, ?, 1)",
                (ref_norm, ref_raw, data_json),
            )
        else:
            conn.execute(
                "UPDATE verses SET ref_raw = ?, request_count = request_count + 1 WHERE ref = ?",
                (ref_raw, ref_norm),
            )


def top_n(n: int = 3):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT ref, data_json, request_count FROM verses ORDER BY request_count DESC, ref ASC LIMIT ?",
            (n,)
        )
        return cur.fetchall()
