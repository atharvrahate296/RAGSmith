"""
RAGSmith – Export service

Packages a project as a self-contained runnable archive.

Export structure:
  /project_name/
    app.py
    faiss.index
    chunks.pkl
    embedding_model/   (optional – skipped if too large; user re-downloads)
    requirements.txt
    README.md
"""

import os
import shutil
import zipfile
import logging
import textwrap
from pathlib import Path

logger = logging.getLogger("ragsmith.export")

EXPORTS_DIR = Path("exports")


def _standalone_app_py(project_name: str, model: str, top_k: int) -> str:
    return textwrap.dedent(
        f'''\
        """
        RAGSmith – Standalone exported RAG app
        Project : {project_name}
        Model   : {model}
        Top-k   : {top_k}

        Run with:
            pip install -r requirements.txt
            python app.py
        Then open http://localhost:8000
        """

        import os, pickle, json, urllib.request, urllib.error
        import numpy as np
        from pathlib import Path
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
        from sentence_transformers import SentenceTransformer
        import faiss

        # ── Config ────────────────────────────────────────────────────────────
        MODEL_NAME   = "{model}"
        TOP_K        = {top_k}
        EMBED_MODEL  = "all-MiniLM-L6-v2"
        OLLAMA_URL   = "http://localhost:11434"

        app = FastAPI(title="RAGSmith – {project_name}")

        # ── Load index & chunks ───────────────────────────────────────────────
        BASE = Path(__file__).parent
        _index  = faiss.read_index(str(BASE / "faiss.index"))
        with open(BASE / "chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)
        _embed_model = SentenceTransformer(EMBED_MODEL)

        # ── Helpers ───────────────────────────────────────────────────────────
        def _search(query: str):
            vec = _embed_model.encode([query], normalize_embeddings=True).astype("float32")
            k   = min(TOP_K, _index.ntotal)
            scores, idxs = _index.search(vec, k)
            results = []
            for score, i in zip(scores[0], idxs[0]):
                if 0 <= i < len(_chunks):
                    results.append((_chunks[i]["text"], float(score), _chunks[i]["filename"]))
            return results

        def _generate(query: str, chunks):
            ctx = "\\n\\n---\\n\\n".join(
                f"[{{fn}} | {{s:.3f}}]\\n{{t}}" for t, s, fn in chunks
            ) or "No context."
            payload = json.dumps({{
                "model": MODEL_NAME,
                "messages": [
                    {{"role": "system", "content": "Answer using ONLY the provided context."}},
                    {{"role": "user",   "content": f"CONTEXT:\\n{{ctx}}\\n\\nQUESTION: {{query}}\\nANSWER:"}},
                ],
                "stream": True,
            }}).encode()
            req = urllib.request.Request(f"{{OLLAMA_URL}}/api/chat", data=payload,
                                         headers={{"Content-Type":"application/json"}}, method="POST")
            parts = []
            with urllib.request.urlopen(req, timeout=120) as r:
                for line in r.read().decode().splitlines():
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        parts.append(obj.get("message", {{}}).get("content", ""))
                        if obj.get("done"): break
                    except Exception:
                        pass
            return "".join(parts).strip()

        # ── API ───────────────────────────────────────────────────────────────
        class Q(BaseModel):
            query: str

        @app.get("/", response_class=HTMLResponse)
        def ui():
            return """<!DOCTYPE html><html><head><title>{project_name} – RAGSmith</title>
            <style>body{{font-family:monospace;max-width:800px;margin:60px auto;padding:20px;}}
            input{{width:100%;padding:10px;font-size:16px;}} button{{padding:10px 20px;cursor:pointer;}}
            #answer{{margin-top:20px;white-space:pre-wrap;background:#f5f5f5;padding:15px;border-radius:6px;}}
            </style></head><body>
            <h1>🔍 {project_name}</h1><p>Powered by RAGSmith</p>
            <input id="q" placeholder="Ask a question…" onkeydown="if(event.key===\\'Enter\\')ask()"/>
            <button onclick="ask()">Ask</button>
            <div id="answer"></div>
            <script>
            async function ask(){{
                const q=document.getElementById("q").value;
                if(!q)return;
                document.getElementById("answer").textContent="Thinking…";
                const r=await fetch("/ask",{{method:"POST",headers:{{"Content-Type":"application/json"}},
                    body:JSON.stringify({{query:q}})}});
                const d=await r.json();
                document.getElementById("answer").textContent=d.answer||d.detail;
            }}
            </script></body></html>"""

        @app.post("/ask")
        def ask(body: Q):
            chunks = _search(body.query)
            answer = _generate(body.query, chunks)
            return {{"query": body.query, "answer": answer,
                    "sources": [{{"text": t[:200], "score": s, "file": f}} for t, s, f in chunks]}}

        if __name__ == "__main__":
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        '''
    )


def _requirements_txt() -> str:
    return (
        "fastapi>=0.110.0\n"
        "uvicorn[standard]>=0.29.0\n"
        "sentence-transformers>=2.7.0\n"
        "faiss-cpu>=1.8.0\n"
        "numpy>=1.26.0\n"
        "pydantic>=2.0.0\n"
    )


def _readme_md(project_name: str, model: str) -> str:
    return textwrap.dedent(
        f"""\
        # {project_name} – Exported RAGSmith Instance

        This is a **self-contained RAG application** exported from RAGSmith.

        ## Requirements
        - Python 3.10+
        - [Ollama](https://ollama.com) running locally with **{model}** pulled:
          ```
          ollama pull {model}
          ```

        ## Quick Start
        ```bash
        pip install -r requirements.txt
        python app.py
        ```
        Then open http://localhost:8000 in your browser.

        ## Files
        | File | Purpose |
        |------|---------|
        | `app.py` | Standalone FastAPI RAG server |
        | `faiss.index` | Vector similarity index |
        | `chunks.pkl` | Document chunk metadata |
        | `requirements.txt` | Python dependencies |

        ## Notes
        - No internet connection required after setup
        - No API keys or subscriptions needed
        - Generated with [RAGSmith](https://github.com/ragsmith/ragsmith) (Open-Source)
        """
    )


def export_project(
    project_id: int,
    project_name: str,
    model: str,
    top_k: int,
) -> str:
    """
    Build a zip archive of the exportable RAG instance.

    Returns
    -------
    str  Path to the created zip file.
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
    build_dir = EXPORTS_DIR / safe_name
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Write app.py
    (build_dir / "app.py").write_text(_standalone_app_py(project_name, model, top_k), encoding="utf-8")

    # Write requirements.txt
    (build_dir / "requirements.txt").write_text(_requirements_txt(), encoding="utf-8")

    # Write README.md
    (build_dir / "README.md").write_text(_readme_md(project_name, model), encoding="utf-8")

    # Copy FAISS index
    idx_src = Path(f"data/indexes/project_{project_id}.index")
    if idx_src.exists():
        shutil.copy(idx_src, build_dir / "faiss.index")
    else:
        logger.warning("No FAISS index found for project %d", project_id)
        # Create empty placeholder
        (build_dir / "faiss.index.missing").write_text(
            "No documents have been indexed for this project yet.", encoding="utf-8"
        )

    # Copy chunks pickle
    pkl_src = Path(f"data/chunks/project_{project_id}.pkl")
    if pkl_src.exists():
        shutil.copy(pkl_src, build_dir / "chunks.pkl")
    else:
        import pickle
        import io
        empty: list = []
        buf = io.BytesIO()
        pickle.dump(empty, buf)
        (build_dir / "chunks.pkl").write_bytes(buf.getvalue())

    # Zip it up
    zip_path = EXPORTS_DIR / f"{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in build_dir.rglob("*"):
            zf.write(file, file.relative_to(EXPORTS_DIR))

    # Cleanup build dir
    shutil.rmtree(build_dir)

    logger.info("Exported project '%s' → %s", project_name, zip_path)
    return str(zip_path)
