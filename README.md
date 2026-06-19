<div align="center">

# 🌊 StateFlow
### Next-Generation Agentic AI Assistant (CRAG + Self-RAG Enabled)

[![Live App](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://stateflow-by-yt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-0052FF?style=for-the-badge&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-F97316?style=for-the-badge)](https://smith.langchain.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-8B5CF6?style=for-the-badge)](https://docs.ragas.io)

<br/>

**StateFlow** is a production-grade, agentic AI assistant built on **LangGraph's state machine architecture**, powered by **Google Gemini 2.5 Flash**, and styled as a pixel-perfect **ChatGPT Dark Mode clone** with advanced self-correcting **Corrective RAG (CRAG)** and **Self-RAG** guardrails.

---

### 🌐 Live Deployment
🚀 **Try the live app here:** **[StateFlow on Streamlit Community Cloud](https://stateflow-by-yt.streamlit.app/)**

---

[🚀 Quick Start](#-setup--installation) · [📸 Demo](#-demo--interface) · [🏗️ Architecture](#️-system-architecture) · [✨ Features](#-features--capabilities) · [📊 RAGAS Evaluation](#-evaluation--ai-quality-ragas)

</div>

## 📸 Demo & Interface

> A pixel-perfect ChatGPT Dark Mode clone — bottom-pinned unified input pill, dynamic sidebar, PDF attachment, and live agent node trace updates.

![StateFlow Demo Screenshot](demo.png)

---

## 🧠 Why LangGraph? The Engineering Rationale

Most candidates build linear chains. StateFlow uses a **cyclic finite-state machine** — here's why that matters:

| Challenge | Traditional Approach | StateFlow (LangGraph) |
|---|---|---|
| **Self-Correction** | Impossible without recursive loops | Cycles state to grade relevancy and rewrite queries when hallucinated |
| **Tool chaining** | Rigid sequential chains | Cyclic graph: LLM → Tool → LLM → Tool (indefinitely) |
| **Session persistence** | Manual DB caching layers | Native `SqliteSaver` / `PostgresSaver` graph checkpointing |
| **Streaming** | Blocking UI transitions | `stream_mode="updates"` feeds active agent states live |
| **State management** | Stateless per-request | Full `TypedDict` state preserved across turns & restarts |

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

### 🛡️ Corrective RAG (CRAG) Integration
- **Document Relevancy Grader**: A dedicated node grades retrieved document chunks for relevance.
- **Web Search Fallback**: If retrieval relevance falls below 50%, the agent automatically triggers a query re-write and falls back to a DuckDuckGo search.
- **Dynamic Query Rewriter**: Uses Gemini to rewrite search terms dynamically to optimize for search engines.

### 🎯 Self-RAG Response Verification
- **Hallucination Grader**: An LLM-based audit node assesses whether the generated response is strictly grounded in the retrieved/search context.
- **Answer Relevancy Grader**: Verifies if the final text directly answers the user's question.
- **Anti-Loop Cap**: Graph state limits retries to a maximum of 3 loops to avoid rate limits or execution stalls.

### 📊 RAGAS Evaluation — AI Quality Measurement
Full evaluation pipeline in [`ragas_evaluation.ipynb`](ragas_evaluation.ipynb) comparing vanilla RAG vs. CRAG + Self-RAG:

| Metric | Baseline RAG | CRAG + Self-RAG |
|---|---|---|
| **Faithfulness** | 0.72 | **0.96** (reduced hallucinations) |
| **Answer Relevancy** | 0.81 | **0.94** (queries rephrased on failure) |
| **Context Recall** | 0.78 | **0.91** (fallback web search backup) |

```bash
jupyter notebook ragas_evaluation.ipynb
```

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
        CRAG[🔍 crag.py\nCorrective RAG Grader & Search]
        SRAG[🎯 srag.py\nSelf-RAG Grounding Auditor]
        RAG[🔍 rag.py\nHybrid RAG Pipeline\nChroma + BM25 Ensemble]
        LLM[🧠 Gemini 2.5 Flash]
    end

    Streamlit <-->|Thread Config + Messages| App
    App <-->|State Snapshots| DB
    App <-->|Invoke + Stream| LLM
    App -->|Evaluates Document Relevance| CRAG
    App -->|Evaluates Answer Utility| SRAG
```

### LangGraph State Machine Flow
```
                   START
                     │
                     ▼
             [route_chat_start]
             /               \
       (RAG Path)         (Legacy Path)
           /                   \
    [retrieve]              [chat_node] ◄──► [tools]
           │
           ▼
    [grade_documents] ────── (Irrelevant) ─────► [rewrite_query] ──► [web_search]
           │                                                               │
      (Relevant)                                                           │
           │                                                               │
           └───────────────────────► [generate] ◄──────────────────────────┘
                                         │
                                         ▼
                               [grade_generation]
                                 /       │      \
                   (Hallucinated)   (Irrelevant)  (Useful)
                       /                 │            \
                 [generate]       [rewrite_query]   [deliver_response] ──► END
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

#### 2. Run the app
```bash
streamlit run frontend/streamlit.py
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini 2.5 Flash | Primary language model |
| **Orchestration** | LangGraph 0.2 | Cyclic agent state machine |
| **Frontend** | Streamlit 1.40 | ChatGPT-clone UI |
| **Vector DB** | ChromaDB 1.5 | Thread registry + PDF embeddings |
| **Evaluation** | RAGAS 0.2 | RAG quality measurement |

---

<div align="center">

Built with ❤️ using **LangGraph** · **ChromaDB** · **Gemini** · **Streamlit**

⭐ Star this repo if it helped you understand production LangGraph architecture!

</div>
