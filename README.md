# StateFlow: LangGraph CRAG & Self-RAG Architecture

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-0052FF?style=for-the-badge&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![PostgreSQL](https://img.shields.io/badge/Checkpointer-PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-F97316?style=for-the-badge)](https://smith.langchain.com)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-8B5CF6?style=for-the-badge)](https://docs.ragas.io)
[![Live App](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://stateflow-by-yt.streamlit.app/)

---

StateFlow is a stateful AI agent built using LangGraph and Google Gemini 2.5 Flash that implements self-correcting retrieval patterns to eliminate hallucinations and resolve the limitations of standard retrieval pipelines.

---

## The Problem

Naive Retrieval-Augmented Generation architectures suffer from systemic failures that prevent deployment in production environments where factual precision is required. Standard top-K vector search retrieves contexts based purely on semantic similarity, which frequently introduces noise, irrelevant document sections, or out-of-context facts that dilute the generation window. When the underlying vector index lacks the target knowledge, naive systems fail silently by forcing the language model to generate answers using sparse or empty context, which directly causes hallucination propagation.

To solve this, retrieval pipelines must transition from rigid linear chains to cyclic agentic networks that evaluate data relevance. An engineered RAG system must verify document relevance before generation and validate factual correctness after generation. Adding corrective query rewrites, external search fallbacks, and multi-cycle self-correction loops ensures that the model only synthesizes responses using verified, relevant context.

---

## Solution Architecture

The state machine is built as a cyclic directed graph using LangGraph to isolate state updates and conditional routing.

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

*Note: The `grade_generation` transition contains an internal loop counter that forces a transition to the END node when `loop_count` reaches 3, preventing execution hangs.*

---

## RAGAS Results

Evaluation of the state machine against 40 complex target QA pairs yielded the following improvements over a baseline RAG configuration:

| Metric | Baseline RAG | CRAG + Self-RAG | Delta |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | `0.72` | **`0.96`** | **`+0.24`** |
| **Context Recall** | `0.78` | **`0.91`** | **`+0.13`** |

Faithfulness increased significantly because the Self-RAG hallucination grader identifies ungrounded assertions and triggers loop cycles to regenerate the output using corrected context. Context Recall improved because the Corrective RAG (CRAG) node detects when retrieved documents are irrelevant and dynamically fetches missing context using external web search fallbacks.

---

## How It Works

### Corrective RAG (CRAG)
CRAG operates as a pre-generation guardrail to verify the quality of the retrieved context. The document relevance grader evaluates each retrieved chunk and calculates a relevance score. If the relevant documents fall below the 50% threshold, the agent routes to a query rewriter to optimize search keywords and executes a web search fallback. This ensures the generation node is never executed using noisy or irrelevant context.

### Self-RAG (Post-Generation Audit)
Self-RAG operates as a post-generation grading loop to audit draft responses. Once the LLM generates a response draft, a hallucination grader assesses if the generated claims are supported by the context documents. Additionally, an answer utility grader determines if the response directly addresses the user query. If either grader returns a negative evaluation, the graph triggers state transitions to rewrite the query or regenerate the draft.

### Hybrid Retrieval Layer
The hybrid retrieval layer combines dense vector search with sparse keyword search. Dense retrieval uses ChromaDB to locate conceptual similarity, while sparse retrieval uses BM25 to capture exact keyword sequences. These two streams are merged using a Reciprocal Rank Fusion (RRF) EnsembleRetriever with weights set to 40% BM25 and 60% semantic search, ensuring optimal recall and precision.

### Enterprise Persistence
Durable session state management is handled using PostgreSQL via the PostgresSaver checkpointer class. This checkpointer serializes the graph state, message history, and tool execution progress to a relational database, allowing threads to survive application restarts. For local testing and development environments where a Postgres database is not configured, the system automatically falls back to a local SQLite checkpointer.

### Observability & Tracing
The application integrates with LangSmith to provide observability into the agentic loops. Every state transition, tool execution, and grader invocation is logged with thread isolation metadata. Developers can inspect prompt inputs, model outputs, token expenses, and execution latency across all graph cycles.

---

## Architecture Decision Records

### Why PostgresSaver over SQLite
PostgresSaver was selected as the primary checkpointer to support concurrent user sessions and transactional database durability. SQLite locks the entire database file during writes, which causes latency bottlenecks and state corruption under parallel execution. PostgreSQL handles row-level locking and persistent connections, which allows StateFlow to scale across multiple container instances while maintaining thread isolation.

### Why hybrid RAG over pure semantic
Pure semantic retrieval using vector embeddings often fails to capture exact keyword queries, code snippets, dates, or product numbers. By combining semantic search with BM25 keyword matching via an EnsembleRetriever, we ensure that both high-level concepts and exact keyword matches are present in the context. The 40/60 weighting provides the best balance, preventing vector noise from drowning out exact keyword targets.

### Why 3-cycle cap specifically
Cyclic graphs run the risk of infinite loops when the model continuously generates ungrounded answers or fails to retrieve relevant data. Capping the loop at 3 cycles prevents excessive API consumption and ensures the system returns a response within a reasonable time window. This threshold provides enough attempts to correct minor formatting or factual errors without causing execution hangs.

### Why asteval over Python eval()
Python's native `eval()` function evaluates arbitrary input strings, exposing the system to Remote Code Execution vulnerabilities. The calculator tool requires a sandboxed interpreter to parse and execute mathematical strings safely. The `asteval` library uses an Abstract Syntax Tree parser that restricts execution to safe mathematical symbols, blocking system builtins, file operations, and imports.

---

## Tool Suite

| Tool | Purpose | Validation | Safety Note |
| :--- | :--- | :--- | :--- |
| `web_search` | Fetches DuckDuckGo snippets. | Pydantic v2 string length check | Max query length set to 100 characters. |
| `calculator` | Evaluates math strings. | Pydantic v2 math expression format | Runs inside an asteval sandbox. |
| `get_stock_price` | Fetches Alpha Vantage quotes. | Pydantic v2 alpha-only ticker check | Limits inputs to 1-5 alphabetic characters. |
| `rag_tool` | Searches indexed PDF context. | Pydantic v2 query and thread ID check | Scope restricted to active thread collection. |

---

## Setup & Installation

### Option 1: Docker Deployment
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Yashthakre-07/StateFlow.git
   cd StateFlow
   ```
2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
3. Populate the secrets inside `.env`:
   ```env
   API_KEY="your-gemini-api-key"
   GEMINI_API_KEY="your-gemini-api-key"
   GOOGLE_API_KEY="your-gemini-api-key"
   ALPHA_VANTAGE_KEY="C9PE94QUEW9VWGFM"
   DB_PATH="chatbot.db"
   POSTGRES_URL="postgresql://postgres:admin_secure_pass@postgres_db:5432/stateflow"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY="your-langsmith-key"
   LANGCHAIN_PROJECT="StateFlow"
   ```
4. Build and start the containers:
   ```bash
   docker-compose up --build
   ```
   The application will run at `http://localhost:8501`.

### Option 2: Local Deployment
1. Install dependencies in your virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your local `.env` file in the root directory.
3. Start the application:
   ```bash
   streamlit run frontend/streamlit.py
   ```

### Option 3: Streamlit Cloud Deployment
1. Deploy the repository using the Streamlit Community Cloud dashboard.
2. Set the main file path to `frontend/streamlit.py`.
3. Configure your API keys and optional database connection strings in the Streamlit Secrets manager.

---

## Running the Test Suite & Evaluation

The unit and integration test suite validates state machine routing, fallback logic, and loop caps using mocked LLM bindings. Execute the test suite with:
```bash
python -m pytest
```
The Ragas evaluation suite is located in `ragas_evaluation.ipynb` and benchmarks generation quality against a 40 QA pair dataset.

---

## Tech Stack Table

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM** | Google Gemini 2.5 Flash | Graph node processing and grader evaluations. |
| **Orchestration** | LangGraph 0.2 | Stateful cyclic directed graphs and routing. |
| **Retrieval** | ChromaDB + BM25 | RRF Ensemble hybrid document search. |
| **Persistence** | PostgreSQL (`PostgresSaver`) | Session transaction checkpointing and state persistence. |
| **Embeddings** | Google Generative AI | High-dimensional text vectorization. |
| **Observability** | LangSmith | Runtime trace analysis and cost tracking. |
| **Evaluation** | RAGAS 0.2 | Quantitative RAG metric benchmarking. |
| **Validation** | Pydantic v2 | Strict validation schemas on tool inputs. |
| **CI/CD** | GitHub Actions + Docker | Automation pipelines and container configurations. |
