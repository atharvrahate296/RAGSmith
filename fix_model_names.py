#!/usr/bin/env python3
"""
RAGSmith – Fix invalid model names in existing projects.

The old UI allowed selecting short model names like 'llama3', 'phi3', 'gemma'
which are valid for Ollama but not for Groq. This script updates any project
using an invalid Groq model name to the correct versioned ID.

Run once:
    python fix_model_names.py
"""

import os, sys

# Load .env
def load_env(path=".env"):
    if not os.path.exists(path): return {}
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.split("#")[0].strip().strip('"').strip("'")
    return env

env = load_env()
DB_DRIVER    = env.get("DB_DRIVER", "sqlite")
DATABASE_URL = env.get("DATABASE_URL", "")
LLM_PROVIDER = env.get("LLM_PROVIDER", "ollama")

# Mapping: old bad name → correct name
GROQ_MODEL_MAP = {
    "llama3":   "llama3-8b-8192",
    "llama2":   "llama3-8b-8192",      # llama2 not on Groq, use llama3
    "mistral":  "mistral:7b",
    "gemma":    "gemma:2b",
}

OLLAMA_MODEL_MAP = {
    # old_shorthand: new_ollama_id
    "mistral": "mistral:7b",
    "gemma": "gemma:2b",
}

model_map = GROQ_MODEL_MAP if LLM_PROVIDER == "groq" else OLLAMA_MODEL_MAP

print(f"RAGSmith – Fix Model Names")
print(f"Provider: {LLM_PROVIDER} | DB: {DB_DRIVER}")
print()

if DB_DRIVER == "postgres":
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        ph   = "%s"
    except Exception as e:
        print(f"Cannot connect to PostgreSQL: {e}")
        sys.exit(1)
else:
    import sqlite3
    conn = sqlite3.connect("data/ragsmith22.db")
    cur  = conn.cursor()
    ph   = "?"

cur.execute("SELECT id, name, model FROM projects ORDER BY id")
rows = cur.fetchall()

if not rows:
    print("No projects found.")
    conn.close()
    sys.exit(0)

fixed = 0
for pid, pname, model in rows:
    if model in model_map:
        new_model = model_map[model]
        cur.execute(f"UPDATE projects SET model={ph} WHERE id={ph}", (new_model, pid))
        print(f"  Fixed project '{pname}' (id={pid}): '{model}' → '{new_model}'")
        fixed += 1
    else:
        print(f"  OK  project '{pname}' (id={pid}): '{model}' (no change needed)")

conn.commit()
conn.close()

print()
print(f"Done — {fixed} project(s) updated.")
