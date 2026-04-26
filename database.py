"""
RAGSmith – Database layer
Supports SQLite (local dev) and PostgreSQL (production/AWS).
Controlled by DB_DRIVER env var via config.py.

SQLite   → uses stdlib sqlite3,  placeholder: ?
Postgres → uses psycopg2,        placeholder: %s
"""

import logging
from pathlib import Path

logger = logging.getLogger("ragsmith.db")


# ── Connection factory ────────────────────────────────────────────────────────

def get_connection():
    """
    Return a DB-API 2.0 connection for the configured driver.
    Rows behave like dicts on both drivers.
    """
    from config import get_settings
    cfg = get_settings()
    if cfg.db_driver == "postgres":
        return _pg_connection(cfg.database_url)
    return _sqlite_connection(cfg.sqlite_path)


def _sqlite_connection(path: str):
    import sqlite3
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _pg_connection(url: str):
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2 not installed. Run: pip install psycopg2-binary"
        ) from exc
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Set DB_DRIVER=sqlite or provide a valid DATABASE_URL for postgres."
        )
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


# ── Placeholder helper ────────────────────────────────────────────────────────

def ph() -> str:
    """Return correct SQL placeholder: '?' for SQLite, '%s' for Postgres."""
    from config import get_settings
    return "%s" if get_settings().db_driver == "postgres" else "?"


# ── Unified execute helpers ───────────────────────────────────────────────────

def db_execute(conn, sql: str, params=(), commit: bool = False):
    """Execute a statement on either driver. Returns cursor."""
    from config import get_settings
    if get_settings().db_driver == "postgres":
        cur = conn.cursor()
        cur.execute(sql, params)
        if commit:
            conn.commit()
        return cur
    else:
        cur = conn.execute(sql, params)
        if commit:
            conn.commit()
        return cur


def db_fetchone(conn, sql: str, params=()):
    cur = db_execute(conn, sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def db_fetchall(conn, sql: str, params=()):
    cur = db_execute(conn, sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_insert(conn, sql: str, params=(), commit: bool = True) -> int:
    """Execute INSERT and return the new row's id."""
    from config import get_settings
    if get_settings().db_driver == "postgres":
        # Append RETURNING id if not present
        if "RETURNING" not in sql.upper():
            sql = sql.rstrip("; ") + " RETURNING id"
        cur = conn.cursor()
        cur.execute(sql, params)
        row_id = cur.fetchone()["id"]
        if commit:
            conn.commit()
        return row_id
    else:
        cur = conn.execute(sql, params)
        if commit:
            conn.commit()
        return cur.lastrowid


# ── Schema ────────────────────────────────────────────────────────────────────

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    DEFAULT '',
    model       TEXT    DEFAULT 'llama-3.1-8b-instant',
    top_k       INTEGER DEFAULT 5,
    created_at  TEXT    DEFAULT (datetime('now')),
    updated_at  TEXT    DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename    TEXT    NOT NULL,
    file_path   TEXT    NOT NULL DEFAULT '',
    num_chunks  INTEGER DEFAULT 0,
    status      TEXT    DEFAULT 'pending',
    error_msg   TEXT    DEFAULT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS query_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    query_text      TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    num_chunks      INTEGER DEFAULT 0,
    grounding_score REAL    DEFAULT 0,
    query_relevance REAL    DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now'))
);
"""

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          SERIAL PRIMARY KEY,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    DEFAULT '',
    model       TEXT    DEFAULT 'llama-3.1-8b-instant',
    top_k       INTEGER DEFAULT 5,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename    TEXT    NOT NULL,
    file_path   TEXT    NOT NULL DEFAULT '',
    num_chunks  INTEGER DEFAULT 0,
    status      TEXT    DEFAULT 'pending',
    error_msg   TEXT    DEFAULT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS query_logs (
    id              SERIAL PRIMARY KEY,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    query_text      TEXT    NOT NULL,
    response        TEXT    NOT NULL,
    num_chunks      INTEGER DEFAULT 0,
    grounding_score REAL    DEFAULT 0,
    query_relevance REAL    DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE OR REPLACE FUNCTION _ragsmith_update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='trg_projects_updated_at') THEN
    CREATE TRIGGER trg_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION _ragsmith_update_updated_at();
  END IF;
END; $$;
"""


# Migration statements — run after schema creation to add new columns
# to existing databases that were created before v2.0.
_SQLITE_MIGRATIONS = [
    "ALTER TABLE query_logs ADD COLUMN grounding_score REAL DEFAULT 0",
    "ALTER TABLE query_logs ADD COLUMN query_relevance REAL DEFAULT 0",
]

_PG_MIGRATIONS = [
    "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS grounding_score REAL DEFAULT 0",
    "ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS query_relevance REAL DEFAULT 0",
]


def init_db() -> None:
    """Create tables if they don't exist, then apply any pending migrations."""
    from config import get_settings
    cfg = get_settings()
    conn = get_connection()
    try:
        if cfg.db_driver == "postgres":
            cur = conn.cursor()
            cur.execute(_PG_SCHEMA)
            for stmt in _PG_MIGRATIONS:
                cur.execute(stmt)
            conn.commit()
            cur.close()
            # Mask password in log
            safe_url = cfg.database_url.split("@")[-1] if "@" in cfg.database_url else cfg.database_url
            logger.info("PostgreSQL initialised at %s", safe_url)
        else:
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()
            # SQLite ALTER TABLE does not support IF NOT EXISTS — use try/except
            for stmt in _SQLITE_MIGRATIONS:
                try:
                    conn.execute(stmt)
                    conn.commit()
                except Exception:
                    pass  # Column already exists — safe to ignore
            logger.info("SQLite initialised at %s", cfg.sqlite_path)
    finally:
        conn.close()
