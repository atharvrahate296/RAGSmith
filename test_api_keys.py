#!/usr/bin/env python3
"""
RAGSmith – API Key & Connectivity Test Script
Run this before starting the server to verify everything is wired up correctly.

Usage:
    python test_api_keys.py
"""

import os
import sys
import json

# ── Load .env manually (no FastAPI/pydantic needed) ──────────────────────────
def load_env(path=".env"):
    env = {}
    if not os.path.exists(path):
        print(f"[WARN] No .env file found at {path}")
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # Strip inline comments
            val = val.split("#")[0].strip().strip('"').strip("'")
            env[key.strip()] = val
    return env

env = load_env()

GROQ_KEY    = env.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
OLLAMA_URL  = env.get("OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
LLM_PROVIDER = env.get("LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "ollama"))
DB_DRIVER   = env.get("DB_DRIVER", os.environ.get("DB_DRIVER", "sqlite"))
DATABASE_URL = env.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))

print("=" * 55)
print("  RAGSmith – API Key & Connectivity Test")
print("=" * 55)
print(f"  LLM_PROVIDER : {LLM_PROVIDER}")
print(f"  DB_DRIVER    : {DB_DRIVER}")
print()

results = []

# ── 1. Python packages ────────────────────────────────────────────────────────
print("── 1. Required packages ─────────────────────────────────")
packages = {
    "fastapi":               "FastAPI web framework",
    "uvicorn":               "ASGI server",
    "requests":              "HTTP client for Groq (Cloudflare-safe)",
    "sentence_transformers": "Embedding model",
    "faiss":                 "Vector store",
    "pydantic_settings":     "Config / .env loading",
    "psycopg2":              "PostgreSQL driver",
    "docx":                  "DOCX parsing (python-docx)",
    "fitz":                  "PDF parsing (PyMuPDF)",
    "boto3":                 "AWS S3 client",
}
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f"  ✓  {pkg:30s} {desc}")
        results.append(("pkg:" + pkg, True))
    except ImportError:
        print(f"  ✗  {pkg:30s} MISSING — pip install {pkg}")
        results.append(("pkg:" + pkg, False))
print()

# ── 2. Groq API ───────────────────────────────────────────────────────────────
print("── 2. Groq API ──────────────────────────────────────────")
if not GROQ_KEY:
    print("  ✗  GROQ_API_KEY not set in .env")
    results.append(("groq_key", False))
else:
    masked = GROQ_KEY[:8] + "..." + GROQ_KEY[-4:]
    print(f"  Key found: {masked}")
    try:
        import requests
        # Test 1: List models
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            timeout=15,
        )
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            print(f"  ✓  Key valid — {len(models)} models available")
            print(f"     Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
            results.append(("groq_key", True))

            # Test 2: Actual completion
            print("  → Testing completion (gemma:2b)…")            
            resp2 = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "gemma:2b",
                    "messages": [{"role": "user", "content": "Reply with exactly: RAGSMITH_OK"}],
                    "max_tokens": 20,
                    "temperature": 0,
                },
                timeout=30,
            )
            resp2.raise_for_status()
            answer = resp2.json()["choices"][0]["message"]["content"].strip()
            tokens = resp2.json().get("usage", {}).get("total_tokens", "?")
            print(f"  ✓  Completion OK — response: '{answer}' ({tokens} tokens)")
            results.append(("groq_completion", True))

        elif resp.status_code == 401:
            print(f"  ✗  Key invalid (401 Unauthorized) — regenerate at console.groq.com")
            results.append(("groq_key", False))
        elif resp.status_code == 403:
            body = resp.text[:200]
            print(f"  ✗  403 Forbidden — Cloudflare block")
            print(f"     Detail: {body}")
            print(f"     This means urllib was used somewhere — requests should fix this")
            results.append(("groq_key", False))
        else:
            print(f"  ✗  Unexpected status {resp.status_code}: {resp.text[:200]}")
            results.append(("groq_key", False))

    except requests.exceptions.ConnectionError as e:
        print(f"  ✗  Network error — cannot reach api.groq.com")
        print(f"     {e}")
        results.append(("groq_key", False))
    except Exception as e:
        print(f"  ✗  Error: {type(e).__name__}: {e}")
        results.append(("groq_key", False))
print()

# ── 3. Ollama ─────────────────────────────────────────────────────────────────
print("── 3. Ollama (local) ────────────────────────────────────")
try:
    import urllib.request
    with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
        data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"  ✓  Ollama running at {OLLAMA_URL}")
        if models:
            print(f"     Models: {', '.join(models)}")
        else:
            print(f"     No models pulled yet. Run: ollama pull mistral")
        results.append(("ollama", True))
except Exception as e:
    print(f"  ✗  Ollama not reachable at {OLLAMA_URL}")
    print(f"     Start with: ollama serve   (or download from https://ollama.com)")
    results.append(("ollama", False))
print()

# ── 4. Database ───────────────────────────────────────────────────────────────
print("── 4. Database ──────────────────────────────────────────")
if DB_DRIVER == "postgres":
    if not DATABASE_URL:
        print("  ✗  DATABASE_URL not set in .env")
        results.append(("db", False))
    else:
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
            conn.close()
            print(f"  ✓  PostgreSQL connected")
            print(f"     {ver[:60]}")
            results.append(("db", True))
        except Exception as e:
            print(f"  ✗  PostgreSQL connection failed: {e}")
            results.append(("db", False))
else:
    print(f"  ✓  SQLite — no setup needed (will create data/ragsmith22.db on first run)")
    results.append(("db", True))
print()

# ── 5. .env sanity checks ─────────────────────────────────────────────────────
print("── 5. .env sanity checks ────────────────────────────────")
checks = [
    ("APP_SECRET_KEY", env.get("APP_SECRET_KEY", ""), "change-me", "Should be changed from default"),
    ("EMBEDDING_MODEL", env.get("EMBEDDING_MODEL", ""), "all-MiniLM-L6-v2", "Must be all-MiniLM-L6-v2"),
]
for name, val, expected, note in checks:
    if not val:
        print(f"  ⚠  {name} not set")
    elif val == expected and name == "APP_SECRET_KEY":
        print(f"  ⚠  {name} = default value — {note}")
    elif val == expected:
        print(f"  ✓  {name} = {val}")
    else:
        print(f"  ✓  {name} = {val}")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 55)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  Result: {passed}/{total} checks passed")

critical_failures = []
if LLM_PROVIDER == "groq" and not any(ok for k, ok in results if k == "groq_key"):
    critical_failures.append("Groq API key invalid or unreachable")
if LLM_PROVIDER == "ollama" and not any(ok for k, ok in results if k == "ollama"):
    critical_failures.append("Ollama not running")
if not any(ok for k, ok in results if k == "db"):
    critical_failures.append("Database not accessible")

if critical_failures:
    print()
    print("  CRITICAL — app will not work:")
    for f in critical_failures:
        print(f"    ✗ {f}")
    print()
    sys.exit(1)
else:
    print()
    print("  ✓  All critical checks passed — safe to start the app")
    print()
    sys.exit(0)
