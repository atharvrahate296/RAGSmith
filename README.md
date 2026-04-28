# 🔍 RAGSmith

> **Fully Open-Source, Self-Hosted RAG (Retrieval-Augmented Generation) Builder**  
> Build, manage, and export custom knowledge bases entirely on your local machine  
> Zero cost • Zero proprietary dependencies • 100% offline

RAGSmith is a complete RAG system that lets you upload documents, automatically chunk and embed them, build vector indexes, and query them using local LLMs—all without leaving your machine.

---

## ✨ Key Features

| Feature | Technology | Details |
|---------|-----------|---------|
| **Multi-Project Management** | FastAPI | Create and manage multiple independent RAG projects |
| **Document Processing** | PyMuPDF, pdfminer, python-docx, etc. | Extract text from PDF, TXT, MD, DOCX, RST, CSV |
| **Embeddings** | SentenceTransformers | Uses `all-MiniLM-L6-v2` for semantic embedding (locally) |
| **Vector Search** | FAISS | Fast similarity search with no internet required |
| **Hybrid Search** | FAISS + BM25 | Combine vector search with keyword matching using Reciprocal Rank Fusion |
| **Local LLM** | Ollama | Mistral, LLaMA 3, Phi-3, Gemma, and more |
| **Cloud LLM Option** | Groq | Optional: Run cloud-based LLMs without API costs |
| **Cross-Encoder Reranking** | sentence-transformers | Re-rank retrieved results for better accuracy |
| **Exportable Projects** | ZIP Archives | Download any project as a self-contained runnable package |
| **Web Interface** | Jinja2 Templates | Clean, intuitive UI for managing projects and queries |
| **Extensible** | Modular Design | Swap embeddings, retrieval methods, or LLMs easily |

---

## 🏗️ Architecture Overview

### How RAGSmith Works

```
1. UPLOAD & PROCESS
   Upload document → Extract text → Chunk into overlapping segments
   
2. EMBED & INDEX
   Embed chunks using all-MiniLM-L6-v2 → Store in FAISS index
   Also build BM25 index for keyword-based retrieval
   
3. STORE
   Save metadata in SQLite/PostgreSQL database
   Store files locally or on S3
   
4. QUERY (RAG Pipeline)
   User query → Embed → Hybrid Search (FAISS + BM25 + RRF)
   → Cross-encoder Re-rank → Generate with LLM → Evaluate response
   
5. EXPORT
   Download entire project as zip with model, config, and data
```

### Tech Stack

- **Backend Framework**: FastAPI
- **Database**: SQLite (default) or PostgreSQL
- **Vector DB**: FAISS (embedded)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Ollama (local) + Groq (optional cloud)
- **Frontend**: Jinja2 templates + vanilla JS
- **Document Processing**: PyMuPDF, pdfminer, python-docx, etc.

---

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (download from [ollama.com](https://ollama.com))
- **Git** (optional, for cloning)

### Optional
- PostgreSQL (if you prefer PostgreSQL over SQLite)
- AWS credentials (if using S3 storage instead of local)
- Groq API key (if you want to use Groq LLMs instead of local Ollama)

---

## 🚀 Quick Start

### 1. Clone or Navigate to the Repository

```bash
cd RAGSmith
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama (Required)

In a separate terminal:
```bash
ollama serve
```

Pull a default model:
```bash
ollama pull mistral
```

**Available models:**
- `mistral` (7B, recommended)
- `llama3` (8B)
- `llama2` (7B)
- `phi3` (3.8B, lightweight)
- `gemma:2b` (2B, very fast)
- `neural-chat` (7B)

### 5. Run RAGSmith

```bash
python main.py
```

The web interface will be available at **http://localhost:8000**

---

## 📁 Project Structure

```
RAGSmithhh/
├── main.py                    # FastAPI application entry point
├── config.py                  # Centralized configuration (env settings)
├── database.py                # Database connection & helpers (SQLite/Postgres)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── routers/                   # API endpoint handlers
│   ├── documents.py           # Document upload & processing endpoints
│   ├── projects.py            # Project CRUD operations
│   ├── query.py               # RAG query pipeline & evaluation
│   ├── sessions.py            # Chat session management
│   ├── export.py              # Project export functionality
│   └── settings.py            # Settings & configuration endpoints
│
├── services/                  # Core business logic
│   ├── processor.py           # Text extraction, chunking, indexing
│   ├── embeddings.py          # Text-to-vector embedding
│   ├── retriever.py           # FAISS search, BM25, RRF
│   ├── reranker.py            # Cross-encoder re-ranking
│   ├── llm.py                 # LLM integration (Ollama + Groq)
│   ├── evaluator.py           # Response evaluation & quality metrics
│   ├── storage.py             # File storage (local/S3)
│   └── exporter.py            # Project packaging & export
│
├── models/                    # Pydantic schemas & request/response models
│   ├── schemas.py             # API models
│   └── __init__.py
│
├── templates/                 # HTML templates
│   └── index.html             # Web UI
│
├── static/                    # Static assets
│   ├── css/
│   │   └── main.css
│   └── js/
│       └── main.js
│
├── data/                      # Runtime data
│   ├── uploads/               # Uploaded documents (organized by project)
│   │   ├── 1/
│   │   ├── 2/
│   │   └── 3/
│   ├── chunks/                # Processed chunks (temporary)
│   ├── indexes/               # FAISS vector indexes
│   │   ├── project_1.index
│   │   ├── project_2.index
│   │   └── project_3.index
│   └── ragsmith22.db          # SQLite database
│
├── exports/                   # Exported projects (zip files)
│
├── env/                       # Python virtual environment
│
├── nginx/                     # Nginx configuration (optional)
│   └── ragsmith.conf
│
└── models/                    # (Placeholder for model files)
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `routers/` | FastAPI route handlers for each domain (documents, projects, queries, etc.) |
| `services/` | Core business logic separated by responsibility (embeddings, retrieval, LLM, etc.) |
| `data/uploads/` | Uploaded documents organized by project ID |
| `data/indexes/` | FAISS vector indexes for each project |
| `templates/` | HTML templates for the web interface |
| `exports/` | Generated ZIP files when exporting projects |

---

## ⚙️ Configuration

All settings are managed in [config.py](config.py) and can be overridden with environment variables or a `.env` file.

### Key Settings

```python
# Application
app_env = "development"  # or "production"
log_level = "INFO"

# Database
db_driver = "sqlite"  # or "postgres"
sqlite_path = "data/ragsmith22.db"

# Embedding Model (always this one)
embedding_model = "all-MiniLM-L6-v2"

# Ollama (Local LLM)
ollama_base_url = "http://localhost:11434"
ollama_default_model = "mistral:7b"
ollama_available_models = "mistral:7b,gemma:2b,llama2:7b,neural-chat:7b"

# Groq (Optional Cloud LLM)
groq_api_key = ""  # Leave empty if not using
groq_default_model = "llama-3.1-8b-instant"

# File Storage
storage_backend = "local"  # or "s3"
local_upload_dir = "data/uploads"
```

### Using PostgreSQL

Create a `.env` file:
```bash
DB_DRIVER=postgres
DATABASE_URL=postgresql://user:password@localhost:5432/ragsmith
```

### Using Groq Cloud LLM

```bash
# Create .env
GROQ_API_KEY=your-groq-api-key-here
```

### Using S3 Storage

```bash
# Create .env
STORAGE_BACKEND=s3
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_S3_BUCKET=your-bucket
AWS_S3_REGION=us-east-1
```

---

## 🎯 Usage Guide

### Via Web Interface (Easiest)

1. Open **http://localhost:8000**
2. Create a new project
3. Upload documents (PDF, TXT, DOCX, etc.)
4. Wait for processing to complete
5. Ask questions about your documents

### Via API

#### Create a Project
```bash
curl -X POST http://localhost:8000/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My RAG", "description": "Test project"}'
```

#### Upload a Document
```bash
curl -X POST http://localhost:8000/api/documents/1 \
  -F "file=@document.pdf"
```

#### Query a Project
```bash
curl -X POST http://localhost:8000/api/query/1 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is mentioned about X?",
    "session_id": "session_1",
    "provider": "ollama"
  }'
```

#### Export a Project
```bash
curl -X POST http://localhost:8000/api/export/1 \
  -H "Content-Type: application/json" \
  --output project.zip
```

---

## 🔧 How It Works (Detailed)

### Document Processing Pipeline

1. **Upload**: Document saved locally or on S3
2. **Extract**: Text extracted using PyMuPDF (PDF), pdfminer (fallback), python-docx (Word), etc.
3. **Chunk**: Text split into 500-char overlapping segments (100-char overlap)
4. **Embed**: Each chunk embedded using `all-MiniLM-L6-v2` (384 dimensions)
5. **Index**: Embeddings stored in FAISS, chunks stored in SQLite
6. **BM25**: Parallel BM25 index built for keyword-based retrieval

### Query Pipeline (RAG)

1. **Embed Query**: User query embedded with same model
2. **Hybrid Retrieval**: 
   - FAISS similarity search (top 20 candidates)
   - BM25 keyword search (top 20 candidates)
   - Reciprocal Rank Fusion combines both rankings
3. **Re-rank**: Cross-encoder model re-scores top candidates
4. **Generate**: Top chunks passed to LLM (Ollama or Groq) with system prompt
5. **Evaluate**: Response quality assessed (token count, relevance, etc.)
6. **Log**: Entire exchange stored for audit trail

---

## 📊 Database Schema (SQLite)

### Projects
```sql
CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);
```

### Documents
```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  filename TEXT,
  num_chunks INTEGER,
  status TEXT,  -- "processing", "ready", "error"
  error_msg TEXT,
  created_at TIMESTAMP
);
```

### Chunks (Text & Embeddings)
```sql
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  document_id INTEGER,
  chunk_index INTEGER,
  text TEXT,
  embedding BLOB  -- 384-dim numpy array
);
```

### Query Logs
```sql
CREATE TABLE query_logs (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  session_id TEXT,
  query TEXT,
  response TEXT,
  model TEXT,
  retrieval_count INTEGER,
  execution_time_ms FLOAT,
  created_at TIMESTAMP
);
```

---

## 🚀 Advanced Features

### Sessions & Chat History

Projects support chat sessions to maintain context across multiple queries.

```bash
curl -X GET http://localhost:8000/api/sessions/1
```

### Response Evaluation

Each response is automatically evaluated for:
- Token efficiency
- Relevance to query
- Hallucination detection
- Answer completeness

### Project Export

Export entire project (model, config, docs, index) as a ZIP:

```bash
curl -X POST http://localhost:8000/api/export/1 --output my_project.zip
```

The ZIP contains everything needed to run queries offline.

### Cross-Encoder Re-ranking

By default, uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-rank retrieved documents for better relevance.

---

## 🔌 Services Overview

| Service | Purpose |
|---------|---------|
| **processor.py** | Text extraction, chunking, and FAISS index building |
| **embeddings.py** | Convert text to vectors using SentenceTransformers |
| **retriever.py** | Search FAISS index and BM25, implement RRF |
| **reranker.py** | Cross-encoder re-ranking of search results |
| **llm.py** | LLM orchestration (Ollama + Groq) |
| **evaluator.py** | Quality metrics and response evaluation |
| **storage.py** | Local/S3 file persistence |
| **exporter.py** | Project packaging and ZIP generation |

---

## 📚 Supported Document Types

| Format | Handler | Notes |
|--------|---------|-------|
| **PDF** | PyMuPDF + pdfminer | Handles scanned PDFs and vector PDFs |
| **TXT** | Plain text | UTF-8 encoding |
| **Markdown** | Markdown parser | Preserves structure |
| **DOCX** | python-docx | Word documents (Office format) |
| **RST** | reStructuredText | Technical documentation format |
| **CSV** | CSV parser | Treats each row as a document |

---

## 🐛 Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve` (in a separate terminal)
- Check: http://localhost:11434/api/tags

### FAISS Index Error
- Delete `data/indexes/` and re-upload documents
- Verify Python version is 3.10+

### Out of Memory
- Reduce `CHUNK_SIZE` in [services/processor.py](services/processor.py)
- Use smaller model in Ollama (e.g., `gemma:2b`)
- Process documents in smaller batches

### Database Locked
- Close other connections to the database
- Check `data/ragsmith22.db` is not open in another application

---

## 📖 API Endpoints

### Projects
- `GET /api/projects` — List all projects
- `POST /api/projects` — Create project
- `GET /api/projects/{id}` — Get project details
- `PUT /api/projects/{id}` — Update project
- `DELETE /api/projects/{id}` — Delete project

### Documents
- `GET /api/documents/{project_id}` — List documents
- `POST /api/documents/{project_id}` — Upload document
- `DELETE /api/documents/{id}` — Delete document

### Queries
- `POST /api/query/{project_id}` — Execute RAG query
- `GET /api/query/{project_id}/logs` — Get query history

### Sessions
- `GET /api/sessions/{project_id}` — List sessions
- `GET /api/sessions/{project_id}/{session_id}` — Get session history

### Export
- `POST /api/export/{project_id}` — Export project as ZIP

### Settings
- `GET /api/settings/models` — List available LLM models

---

## 🛠️ Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black . && isort .
```

### Type Checking
```bash
mypy routers/ services/ models/
```

---

## 📄 License

RAGSmith uses open-source components with their respective licenses:
- **FAISS** — MIT
- **Ollama** — MIT
- **SentenceTransformers** — Apache 2.0
- **FastAPI** — MIT
- **PyMuPDF** — AGPL (commercial license available)

Check individual component licenses in [requirements.txt](requirements.txt).

---

## 🤝 Contributing

Contributions welcome! Some areas for improvement:

- [ ] Add GraphQL API alternative
- [ ] Implement WebSocket support for streaming responses
- [ ] Add fine-tuning pipeline for custom models
- [ ] Support for video/audio documents
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Unit & integration tests

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review [config.py](config.py) for configuration options
3. Check application logs: `log_level = "DEBUG"` in [config.py](config.py)

---

## 🎓 Learn More

- [FAISS Documentation](https://faiss.ai/)
- [Sentence-Transformers](https://huggingface.co/sentence-transformers/)
- [Ollama Models](https://ollama.ai/library)
- [FastAPI](https://fastapi.tiangolo.com/)
- [RAG Concepts](https://en.wikipedia.org/wiki/Prompt_engineering#Retrieval-augmented_generation)

---

**Happy building! 🚀**
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
