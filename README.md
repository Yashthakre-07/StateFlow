# StateFlow: LangGraph CRAG & Self-RAG Architecture

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-0052FF?style=for-the-badge&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![PostgreSQL](https://img.shields.io/badge/Checkpointer-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-F97316?style=for-the-badge)](https://smith.langchain.com)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-8B5CF6?style=for-the-badge)](https://docs.ragas.io)

---

## The Problem with Naive RAG

Naive Retrieval-Augmented Generation (RAG) pipelines suffer from three fundamental points of failure:
1. **Low Retrieval Precision**: Naive systems retrieve documents based on superficial semantic similarity, often injecting irrelevant context chunks into the prompt.
2. **Hallucination Propagation**: If retrieved contexts contain gaps or incorrect facts, the LLM generates fabricated answers unsupported by the source material.
3. **Out-of-Domain Failure**: When the queried information is completely missing from the internal knowledge base, the pipeline fails silently or returns incorrect details rather than sourcing external data.

**StateFlow** resolves these failure modes by transitioning the linear RAG pipeline into a cyclic, stateful agent loop. It continuously grades retrieved document relevance, dynamically rewrites queries to search the web for missing information, and audits generated answers for hallucinations before they are served to the user.

---

## System Architecture

StateFlow uses a cyclic state machine built on **LangGraph**. The workflow isolates retrieval, assessment, generation, and correction into distinct executable steps:

### Graph State Machine Flow

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

## RAGAS Evaluation Results

A comparative evaluation was executed across 40 complex QA pairs. Below is the performance uplift comparing a baseline RAG pipeline to the StateFlow (CRAG + Self-RAG) agent:

| Metric | Baseline RAG | CRAG + Self-RAG Agent | Target Improvement Area |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | `0.72` | **`0.96`** | Hallucination mitigation and fact-grounding. |
| **Answer Relevancy** | `0.81` | **`0.94`** | Minimizing off-topic and incomplete responses. |
| **Context Recall** | `0.78` | **`0.91`** | Retrieving missing context via search fallbacks. |

---

## Why This Matters: CRAG vs. Self-RAG

Building a reliable RAG agent requires separating **retrieval verification** from **generation verification**. StateFlow maintains this boundary by implementing two distinct grading guardrails:

### Corrective RAG (CRAG)
CRAG operates **pre-generation**. It evaluates the retrieved documents for query relevance. If less than 50% of the retrieved contexts contain answers to the query, CRAG halts generation, rephrases the search terms using a query rewriter, and executes a web search (DuckDuckGo). This ensures that the generation node is never fed irrelevant or empty context.

### Self-RAG
Self-RAG operates **post-generation**. Once the LLM generates a response draft, Self-RAG runs a double-audit check:
1. **Hallucination Grade**: It verifies if all claims in the response are fully grounded in the retrieved documents. If ungrounded, it forces a regeneration.
2. **Answer Utility Grade**: It verifies if the generated text directly answers the user's question. If not, it triggers a query rewrite to re-retrieve new context.

---

## Setup & Installation

### Option A: Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/Yashthakre-07/StateFlow.git
   cd StateFlow
   ```
2. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Provide your Gemini API keys and set your PostgreSQL URL:
   ```env
   API_KEY="your-gemini-key"
   POSTGRES_URL="postgresql://postgres:admin_secure_pass@postgres_db:5432/stateflow"
   ```
3. Build and spin up the containers:
   ```bash
   docker-compose up --build
   ```
   The application will be accessible at `http://localhost:8501`.

---

### Option B: Local Development

1. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your local environment secrets in a `.env` file in the project root.
3. Start the application:
   ```bash
   streamlit run frontend/streamlit.py
   ```

---

### Option C: Streamlit Community Cloud

This project is optimized for deployment to Streamlit Community Cloud:
1. Link your GitHub repository to Streamlit Community Cloud.
2. Set your main entry file path to `frontend/streamlit.py`.
3. Under **Advanced settings -> Secrets**, supply your Gemini keys (`API_KEY`) and optional `POSTGRES_URL` connection strings.

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM** | Google Gemini 2.5 Flash | Core reasoning and grading engine. |
| **Orchestrator** | LangGraph 0.2 | Stateful agentic cyclic routing. |
| **Checkpointer** | PostgreSQL (`PostgresSaver`) | Durable agent session state persistence. |
| **Vector Database** | ChromaDB 1.5 | Sparse document storage and thread registries. |
| **User Interface** | Streamlit 1.40 | Live agent node visualization portal. |
| **Evaluation** | RAGAS 0.2 | Factual metrics benchmarking. |

---

## Architecture Decision Records (ADRs)

### ADR 001: Hybrid RAG (Dense + Sparse Search)
*   **Context**: Dense semantic searches (ChromaDB) capture general concepts but miss exact keyword hits, code syntax, or timestamps.
*   **Decision**: We implemented an `EnsembleRetriever` combining ChromaDB semantic search with a BM25 sparse keyword retriever (40% sparse / 60% dense weights). This combination ensures high recall and precise keyword targeting.

### ADR 002: PostgresSaver Checkpointing
*   **Context**: SQLite is simple but lacks parallel session concurrency, schema migrations, and high availability needed in production.
*   **Decision**: We configured `PostgresSaver` as the primary persistence layer. It safely handles binary serialization of message states, allowing secure thread isolation and horizontal scaling.

### ADR 003: 3-Cycle Safety Cap
*   **Context**: Agentic loops run the risk of infinite loops (hallucinating, rewriting, and retrying endlessly) if context is extremely sparse.
*   **Decision**: We configured a loop counter in the `ChatState` schema, capping iterations at 3. When `loop_count >= 3`, the graph routes to END, returning the best available response.

### ADR 004: asteval Sandbox for Math Calculations
*   **Context**: Using Python's native `eval()` exposes the application to Remote Code Execution (RCE) vulnerabilities.
*   **Decision**: The calculator tool uses the `asteval` library to parse mathematical expressions inside a sandboxed interpreter, blocking imports, builtins, and system calls.
