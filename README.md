<div align="center">

# 🌊 StateFlow
### Next-Generation Agentic AI Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-0052FF?style=for-the-badge&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-F97316?style=for-the-badge)](https://smith.langchain.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-8B5CF6?style=for-the-badge)](https://docs.ragas.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

**StateFlow** is a production-grade, agentic AI assistant built on **LangGraph's finite-state machine architecture**, powered by **Google Gemini 2.5 Flash**, and styled as a pixel-perfect **ChatGPT Dark Mode clone** — all in Streamlit.

It features **Hybrid RAG** (ChromaDB + BM25 ensemble), **RAGAS evaluation**, **MCP tool integrations**, **LangSmith tracing**, **live token economics**, and a **fully Dockerized deployment pipeline**.

[🚀 Quick Start](#-setup--installation) · [📸 Demo](#-demo--interface) · [🏗️ Architecture](#️-system-architecture) · [✨ Features](#-features--capabilities) · [📊 RAGAS Evaluation](#-evaluation--ai-quality-ragas)

</div>

---

## 📸 Demo & Interface

> A pixel-perfect ChatGPT Dark Mode clone — bottom-pinned unified input pill, dynamic sidebar, PDF attachment, and live token economics tracker.

![StateFlow Demo Screenshot](demo.png)

---

## 🧠 Why LangGraph? The Engineering Rationale

Most candidates build linear chains. StateFlow uses a **cyclic finite-state machine** — here's why that matters:

| Challenge | Traditional Approach | StateFlow (LangGraph) |
|---|---|---|
| **Tool chaining** | Rigid sequential chains | Cyclic graph: LLM → Tool → LLM → Tool (indefinitely) |
| **Session persistence** | Manual DB caching layers | Native `SqliteSaver` / `PostgresSaver` graph checkpointing |
| **Streaming** | Blocking UI transitions | `stream_mode="messages"` streams tokens + tool statuses live |
| **State management** | Stateless per-request | Full `TypedDict` state preserved across turns & restarts |
| **Routing** | Hard-coded `if/else` | `tools_condition` conditional edges — LLM decides the path |

---

## ✨ Features & Capabilities

### 🎨 Premium UI — ChatGPT Dark Mode Clone
- **Radial gradient dark background** with `#0c0d12` deep space tone
- **Outfit Google Font** — 300–800 weight range, letter-spacing tuned
- **Bottom-pinned ChatGPT pill input** — `position: fixed`, columns merged into a single bar at viewport bottom
- **Circular `+` upload button** — Streamlit's file uploader completely restyled via CSS pseudo-elements into a transparent `+` icon
- **Glassmorphism chat bubbles** — `backdrop-filter: blur(8px)`, hover lift animations
- **Sidebar thread history** — Auto-named from first user message, defaults to `💬 New Chat`
- **Purple-blue gradient header** — `linear-gradient(90deg, #a78bfa, #3b82f6, #f472b6)`
- **Responsive media queries** — sidebar-aware centering across breakpoints

### 🗄️ ChromaDB — Dual-Purpose Primary Database
- **Thread Registry Collection** (`thread_registry`): Every conversation thread is registered in ChromaDB with `username`, `created_at` timestamp, and `title` metadata — ChromaDB is the source of truth for all session data
- **Vector Collections** (per thread): Each uploaded PDF creates a dedicated Chroma collection (`thread_<id>`) for isolated retrieval
- **Persistent local storage** at `./chroma_db` — survives restarts without re-ingestion
- **HTTP remote mode** supported via `CHROMA_HOST` env var for production cloud deployments

### 🔍 Hybrid RAG — Ensemble Retrieval Pipeline
```
Query ──► ChromaDB (Dense Semantic Search, k=4)  ──┐
       └► BM25 (Sparse Keyword Search, k=4)       ──┼──► EnsembleRetriever (0.4/0.6) ──► Context
                                                      └── Reciprocal Rank Fusion (RRF)
```
- **ChromaDB dense retrieval** — cosine similarity over `models/embedding-001` (Google) or `text-embedding-3-small` (OpenAI)
- **BM25 sparse retrieval** — `langchain_community.retrievers.BM25Retriever`, dynamically built from Chroma's stored documents
- **EnsembleRetriever** — 40% BM25 weight + 60% semantic weight for optimal precision-recall balance
- **Graceful fallback** — falls back to pure Chroma if BM25 dependencies fail
- **Pydantic RAGInput schema** — validates query length (max 200 chars) and strips whitespace before retrieval

### 📊 RAGAS Evaluation — AI Quality Measurement
> *"Most candidates don't measure AI quality — they just build it."*

Full evaluation pipeline in [`ragas_evaluation.ipynb`](ragas_evaluation.ipynb) measuring:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Is the answer grounded strictly in the retrieved context? |
| **Answer Relevancy** | Does the response address the user's actual question? |
| **Context Recall** | Do retrieved chunks contain all necessary ground-truth facts? |
| **Context Precision** | Are retrieved chunks ranked correctly (relevant ones first)? |

```bash
jupyter notebook ragas_evaluation.ipynb
```

### 🛠️ Agentic Tool Suite
| Tool | Description | Validation |
|---|---|---|
| 🔍 `web_search` | DuckDuckGo live search (US-EN region) | `SearchInput` — max 100 chars |
| 🧮 `calculator` | Sandboxed `asteval` math (no `eval()` RCE risk) | `CalculatorInput` — max 200 chars |
| 📈 `get_stock_price` | Alpha Vantage `GLOBAL_QUOTE` real-time prices | `StockInput` — alpha-only 1–5 char ticker |
| 📄 `rag_tool` | Hybrid RAG search over uploaded PDF | `RAGInput` — thread-scoped |
| 🔌 MCP Tools | External `MultiServerMCPClient` integrations | Dynamic load via `langchain_mcp_adapters` |

All tools use **Pydantic input schemas** with `@field_validator` — invalid inputs are rejected before any API call is made.

### 📡 LangSmith Observability & Tracing
Every agent execution is automatically traced with:
```python
CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key, "user_action": "chat_turn"},
    "tags": [f"thread_id:{thread_key}", "user_action:chat_turn"],
    "run_name": "chat_turn",
}
```
- Full token counts, tool call sequences, latency, and cost per run
- Searchable by `thread_id` tag in LangSmith dashboard
- Project name: `StateFlow`

### 💰 Live Token Economics Sidebar
- **Real-time cumulative tracking** of input + output tokens per thread
- **Gemini 2.5 Flash pricing**: `$0.075/1M input`, `$0.30/1M output`
- **$2.00 per-conversation billing cap** — hard stop with user-facing warning
- Metrics persist in `st.session_state["token_tracker"]` across page rerenders

### 🔒 Security & Safety
- **`asteval` sandbox** — replaces `eval()` for calculator to eliminate RCE attack surface
- **Pydantic validators** on all tool inputs — type-safe, length-bounded, stripped of whitespace
- **`.env` secrets** — never committed (in `.gitignore`), loaded via `pydantic-settings`
- **Thread isolation** — each user's threads are namespaced under `{username}_` prefix in both ChromaDB and SqliteSaver

### 🚀 CI/CD & Deployment
- **Dockerfile** — single-stage Python image with `CMD streamlit run`
- **docker-compose.yml** — orchestrated multi-container with host-mounted `chroma_db/` and `chatbot.db` volumes
- **GitHub Actions** (`ci.yml`) — auto-runs on push/PR to validate build and dependencies

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([👤 User]) <--> Streamlit[🖥️ Streamlit UI\nChatGPT Dark Mode Clone]

    subgraph Frontend
        Streamlit
    end

    subgraph Backend
        App[⚙️ app.py\nLangGraph Orchestrator]
        DB[🗄️ database.py\nChromaDB Thread Registry\n+ SqliteSaver Checkpointer]
        Tools[🛠️ tools.py\nSearch · Calculator · Stock]
        RAG[🔍 rag.py\nHybrid RAG Pipeline\nChroma + BM25 Ensemble]
        MCP[🔌 mcp.py\nMCP Client Integrations]
        LLM[🧠 Gemini 2.5 Flash]
        LS[📡 LangSmith Tracing]
    end

    Streamlit <-->|Thread Config + Messages| App
    App <-->|State Snapshots| DB
    App <-->|Invoke + Stream| LLM
    LLM <-->|Tool Calls| Tools
    LLM <-->|RAG Query| RAG
    LLM <-->|External Tools| MCP
    App -->|Traces + Metrics| LS
```

### LangGraph State Machine Flow
```
START
  │
  ▼
[chat_node] ──── tools_condition ────► [tools] ──┐
  ▲                                               │
  └───────────────────────────────────────────────┘
  │
  ▼ (no more tool calls)
END
```

### Execution Flow
1. **User Input** → Custom-styled bottom-pinned pill input in Streamlit
2. **Thread Lookup** → ChromaDB `thread_registry` resolves session; SqliteSaver restores graph state
3. **LangGraph Streaming** → `chatbot.stream(..., stream_mode="messages")` emits token-by-token
4. **Tool Execution** → `ToolNode` dispatches to web search / calculator / stock / RAG / MCP
5. **Hybrid RAG** → BM25 + ChromaDB ensemble retriever returns ranked context chunks
6. **Response Streaming** → `st.write_stream()` renders tokens live; `st.status()` shows tool activity
7. **State Persistence** → Graph checkpointed to SQLite; thread metadata updated in ChromaDB
8. **Token Tracking** → Input/output counts accumulated in `st.session_state`, sidebar updates

---

## 📁 Project Structure

```
StateFlow/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── .env.example                    # Environment variable template
├── .gitignore                      # Excludes secrets, DBs, caches
├── Dockerfile                      # Container build configuration
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Pinned Python dependencies
├── demo.png                        # Interface screenshot
├── ragas_evaluation.ipynb          # 📊 RAGAS AI quality evaluation notebook
│
├── backend/
│   ├── __init__.py
│   ├── app.py                      # LangGraph graph compiler & agent orchestrator
│   ├── config.py                   # Pydantic-settings config loader (.env)
│   ├── database.py                 # ChromaDB thread registry + SqliteSaver checkpointer
│   ├── rag.py                      # Hybrid RAG: ChromaDB + BM25 EnsembleRetriever
│   ├── tools.py                    # Web search, calculator, stock price tools
│   └── mcp.py                      # Model Context Protocol multi-server client
│
└── frontend/
    └── streamlit.py                # ChatGPT-clone Streamlit UI (549 lines of premium CSS + logic)
```

---

## 🚀 Setup & Installation

### Option A — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/Yashthakre-07/StateFlow.git
cd StateFlow

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Build & launch
docker-compose up --build
```

App runs at **http://localhost:8501** with persistent volumes for `chroma_db/` and `chatbot.db`.

---

### Option B — Local Development

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure environment variables
Create a `.env` file in the project root:
```env
# Required
API_KEY="your-gemini-api-key"

# LangSmith Tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="StateFlow"

# Optional: Use OpenAI embeddings instead of Google
# OPENAI_API_KEY="your-openai-api-key"

# Optional: Remote ChromaDB server
# CHROMA_HOST="localhost"
# CHROMA_PORT=8000

# Optional: PostgreSQL for production checkpointing
# POSTGRES_URL="postgresql://user:pass@host:5432/db"

# Optional: Alpha Vantage for stock prices
# ALPHA_VANTAGE_KEY="your-key"
```

#### 3. Run the app
```bash
streamlit run frontend/streamlit.py
```

---

## 🧪 Running RAGAS Evaluation

```bash
# Install evaluation extras (already in requirements.txt)
pip install ragas jupyter

# Launch the evaluation notebook
jupyter notebook ragas_evaluation.ipynb
```

The notebook walks through:
- Building a test dataset of question/answer/context triplets
- Running RAGAS scoring (faithfulness, answer_relevancy, context_recall, context_precision)
- Interpreting results to tune retrieval parameters

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini 2.5 Flash | Primary language model |
| **Orchestration** | LangGraph 0.2 | Cyclic agent state machine |
| **Frontend** | Streamlit 1.40 | ChatGPT-clone UI |
| **Vector DB** | ChromaDB 1.5 | Thread registry + PDF embeddings |
| **Sparse Retrieval** | BM25 (rank_bm25) | Keyword-based document retrieval |
| **Embeddings** | Google `embedding-001` / OpenAI `text-embedding-3-small` | Text vectorization |
| **Checkpointing** | SqliteSaver / PostgresSaver | LangGraph state persistence |
| **Observability** | LangSmith | Traces, metrics, token costs |
| **Evaluation** | RAGAS 0.2 | RAG quality measurement |
| **Tool Safety** | asteval | Sandboxed math expression evaluation |
| **Validation** | Pydantic v2 | Input schema validation for all tools |
| **MCP** | langchain-mcp-adapters | Model Context Protocol tool loading |
| **CI/CD** | GitHub Actions + Docker | Build validation & containerized deployment |

---

## 💡 Key Design Decisions

### 1. Why ChromaDB as the Primary Database?
ChromaDB serves **two roles** in StateFlow:
- **Thread Registry**: A `thread_registry` Chroma collection tracks all conversation sessions with username, timestamps, and auto-generated titles — replacing SQL queries for thread discovery
- **Vector Store**: Per-thread collections (`thread_<uuid>`) store PDF chunk embeddings for semantic retrieval

This unifies the storage layer under a single persistent client (`./chroma_db/`).

### 2. Why Keep SqliteSaver alongside ChromaDB?
LangGraph's checkpointer uses a proprietary binary serialization format for graph state snapshots (message histories, tool call sequences, node states). ChromaDB is a vector database — it cannot store arbitrary binary blobs in LangGraph's format. SqliteSaver handles this internal layer transparently.

### 3. Why Hybrid RAG over Pure Semantic Search?
Dense embeddings (ChromaDB) excel at *semantic similarity* but can miss exact keyword matches (e.g., specific names, codes, dates). BM25 excels at *keyword precision* but misses synonyms and paraphrases. The **40/60 ensemble** captures both, consistently outperforming either alone on the RAGAS benchmark.

### 4. Why `asteval` instead of Python `eval()`?
Python's `eval()` executes arbitrary code — a critical RCE (Remote Code Execution) vulnerability in production AI systems. `asteval` parses only mathematical expressions in a sandboxed interpreter with no access to builtins, filesystem, or network.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

<div align="center">

Built with ❤️ using **LangGraph** · **ChromaDB** · **Gemini** · **Streamlit**

⭐ Star this repo if it helped you understand production LangGraph architecture!

</div>
