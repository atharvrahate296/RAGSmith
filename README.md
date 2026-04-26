# ⬡ RAGSmith

> **Fully Open-Source, Self-Hosted RAG Builder with Local Export**  
> Version 1.0 · Zero proprietary dependencies · Zero cost

RAGSmith lets you build, manage, and export custom **Retrieval-Augmented Generation (RAG)** pipelines entirely on your local machine — no OpenAI, no Pinecone, no subscriptions.

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| **Multi-Project** | Manage independent RAG knowledge bases |
| **Local Embeddings** | `all-MiniLM-L6-v2` via SentenceTransformers (Apache 2.0) |
| **Vector Search** | FAISS (MIT) — runs 100% offline |
| **Local LLM** | Ollama (MIT) — Mistral, LLaMA 3, Phi-3 |
| **Exportable** | Download any project as a self-contained runnable zip |
| **Zero Cost** | No paid APIs, no cloud, no usage limits |
| **Document Types** | PDF, TXT, MD, DOCX, RST, CSV |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull a local model (Ollama)

```bash
# Default: Mistral 7B (Apache 2.0)
ollama pull mistral

# Alternatives:
# ollama pull llama3
# ollama pull phi3
```

### 4. Run RAGSmith

```bash
python -m uvicorn main:app --reload --host localhost --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## 📐 Architecture

```
Upload → Extract → Chunk → Embed → FAISS Index → Store
                                              ↓
Query  → Embed → FAISS Search → Context → Ollama LLM → Answer
```

### Stack (100% Open-Source)

| Layer | Technology | License |
|-------|-----------|---------|
| Backend | FastAPI | MIT |
| Runtime | Python 3.10+ | PSF |
| Database | SQLite | Public Domain |
| Embeddings | SentenceTransformers | Apache 2.0 |
| Embedding Model | all-MiniLM-L6-v2 | Apache 2.0 |
| Vector Store | FAISS | MIT |
| LLM Runtime | Ollama | MIT |
| LLM Model | Mistral 7B | Apache 2.0 |
| LLM Model | LLaMA 3 | Meta (Free) |
| LLM Model | Phi-3 | MIT |
| Frontend | HTML5 / CSS3 / JS | Web Standards |

---

## 📁 Project Structure

```
ragsmith/
├── main.py                # FastAPI app & lifespan
├── database.py            # SQLite init & connection
├── requirements.txt       # All Python dependencies
├── models/
│   └── schemas.py         # Pydantic request/response models
├── routers/
│   ├── projects.py        # Project CRUD
│   ├── documents.py       # File upload & processing
│   ├── query.py           # RAG pipeline endpoint
│   └── export.py          # Standalone export
├── services/
│   ├── embeddings.py      # SentenceTransformers wrapper
│   ├── processor.py       # Extract → chunk → embed → FAISS
│   ├── llm.py             # Ollama local LLM interface
│   └── exporter.py        # Zip export builder
├── static/
│   ├── css/main.css       # Dark terminal UI styles
│   └── js/main.js         # Vanilla JS SPA logic
├── templates/
│   └── index.html         # Main HTML template
└── data/                  # Auto-created at runtime
    ├── ragsmith22.db        # SQLite database
    ├── indexes/           # FAISS .index files
    ├── chunks/            # Chunk metadata pickles
    └── uploads/           # Original uploaded files
```

---

## 🔌 API Reference

Full interactive docs at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/projects/` | List all projects |
| `POST` | `/api/projects/` | Create project |
| `DELETE` | `/api/projects/{id}` | Delete project |
| `GET` | `/api/documents/{pid}` | List documents |
| `POST` | `/api/documents/{pid}/upload` | Upload & process file |
| `DELETE` | `/api/documents/{pid}/doc/{did}` | Delete document |
| `POST` | `/api/query/{pid}` | Run RAG query |
| `GET` | `/api/query/{pid}/history` | Query history |
| `GET` | `/api/export/{pid}` | Download project as zip |

---

## 📦 Export Format

```
project_name/
├── app.py              # Standalone FastAPI RAG server
├── faiss.index         # Vector similarity index
├── chunks.pkl          # Document chunk metadata
├── requirements.txt    # Minimal dependencies
└── README.md           # Usage instructions
```

Run exported instance:

```bash
pip install -r requirements.txt
ollama pull mistral
python app.py
# → http://localhost:8000
```

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |

Edit `services/llm.py` to change `OLLAMA_BASE_URL` or model defaults.

---

## 📚 Academic Relevance

RAGSmith demonstrates mastery of:
- Applied Natural Language Processing
- Vector representation learning & embedding spaces
- Information retrieval (FAISS kNN search)
- LLM integration and local inference
- AI system orchestration and pipeline design
- Efficient local AI deployment without cloud dependency

---

## 📄 License

RAGSmith is released under the **MIT License**.  
All dependencies use permissive open-source licenses (MIT, Apache 2.0, BSD, Public Domain).

---

*RAGSmith — Build your knowledge, own your AI.*
