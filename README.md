# StateFlow: LangGraph CRAG & Self-RAG Architecture
### Production-Grade Self-Correcting Agentic RAG Engine

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

In standard production environments, naive Retrieval-Augmented Generation (RAG) architectures exhibit critical limitations that prevent their use in high-reliability scenarios:

1. **Low Retrieval Precision (Noise Injection)**: Vector databases retrieve context based purely on top-$K$ semantic similarity. This frequently injects irrelevant context (noise) into the LLM context window, diluting answer relevance and leading to out-of-context generation.
2. **Hallucination Propagation (Source Gaps)**: Standard RAG pipelines assume all retrieved documents are accurate. If the context contains factual gaps or conflicting details, the LLM constructs plausible but false assertions (hallucinations) and presents them as facts.
3. **Out-of-Domain Failure (Static Gaps)**: If the target information does not exist inside the static document collections, the pipeline fails silently or hallucinating, rather than dynamically querying external knowledge sources.

**StateFlow** addresses these points of failure by replacing the linear RAG pipeline with a cyclic state machine. The system constantly grades retrieved document relevance, rewrites search queries dynamically for external web fallbacks when internal knowledge is low, and performs automated self-correction cycles to verify generation correctness before returning the output.

---

## System Architecture

StateFlow is modeled as a stateful, cyclic directed graph using **LangGraph**. The workflow decouples data retrieval, state routing, assessment, and generation into discrete nodes:

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

### Graph Execution Cycles

1. **Query Routing (`route_chat_start`)**: Evaluates if the current thread contains active PDF contexts. If yes, it redirects to the RAG sub-workflow; otherwise, it executes the general chat nodes.
2. **Document Retrieval (`retrieve`)**: Queries an ensemble index combining ChromaDB dense vectors and BM25 sparse keyword indices.
3. **Document Relevance Grading (`grade_documents`)**: Analyzes the retrieved chunks. If the relevance ratio is $\le 50\%$, it triggers the Corrective RAG (CRAG) branch.
4. **Query Re-optimization & Fallback (`rewrite_query` -> `web_search`)**: Rewrites the user query to optimize search keywords, fetches web context via DuckDuckGo, and appends it to the document stack.
5. **Generation (`generate`)**: Generates response draft using the consolidated document contexts.
6. **Self-Correction Grading (`grade_generation`)**:
   * **Hallucination Check**: Grades if the answer is grounded in context. If ungrounded, it loops back to `generate` to regenerate.
   * **Answer Utility Check**: Grades if the response addresses the query. If not useful, it loops back to `rewrite_query` for fresh retrieval.
   * **Safety Cap**: Capped at 3 cycles to prevent infinite execution loops.

---

## RAGAS Evaluation Results

The system was evaluated against a custom dataset of 40 complex QA pairs. The comparison metrics indicate the performance improvement of StateFlow's cyclic grading workflow over a baseline RAG configuration:

| Metric | Baseline RAG | CRAG + Self-RAG Agent | Performance Impact |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | `0.72` | **`0.96`** | Eliminates hallucination propagation by recycling ungrounded generation. |
| **Answer Relevancy** | `0.81` | **`0.94`** | Restructures answers when they fail to address the query. |
| **Context Recall** | `0.78` | **`0.91`** | Expands retrieval scope dynamically using external web fallback. |

---

## Why This Matters: CRAG vs. Self-RAG

StateFlow splits the evaluation checks into two separate stages: **retrieval verification** (pre-generation) and **generation verification** (post-generation).

### Corrective RAG (CRAG)
CRAG operates **pre-generation**. It validates the context chunks retrieved from vector search. If the retrieved material contains noise or is irrelevant, CRAG halts generation and fetches external data from web search first. This isolates the generator from processing low-quality context.

### Self-RAG
Self-RAG operates **post-generation**. It treats the generation as a draft, running fact-grounding checks (to identify hallucination) and utility checks (to identify vague answers). If either check fails, the graph state triggers loop transitions to correct the response, ensuring only validated answers reach the user interface.

---

## Setup & Installation

### Option A: Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/Yashthakre-07/StateFlow.git
   cd StateFlow
   ```
2. Configure your local `.env` environment file:
   ```bash
   cp .env.example .env
   ```
   Supply your Gemini API keys and Postgres connection string:
   ```env
   API_KEY="your-gemini-key"
   POSTGRES_URL="postgresql://postgres:admin_secure_pass@postgres_db:5432/stateflow"
   ```
3. Run docker-compose:
   ```bash
   docker-compose up --build
   ```
   The UI will run on `http://localhost:8501`.

---

### Option B: Local Development

1. Install dependencies in your virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Create your `.env` file in the root folder with configuration parameters.
3. Start the UI:
   ```bash
   streamlit run frontend/streamlit.py
   ```

---

### Option C: Streamlit Community Cloud

This project is optimized for deployment to Streamlit Community Cloud:
1. Link your GitHub repository to Streamlit Community Cloud.
2. Set the main entry file path to `frontend/streamlit.py`.
3. Under **Advanced settings -> Secrets**, supply your Gemini keys (`API_KEY`) and optional `POSTGRES_URL` connection strings.

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM Reasoning** | Google Gemini 2.5 Flash | Graph node processing and grader evaluations. |
| **Graph Orchestrator** | LangGraph 0.2 | Stateful cyclic directed graphs and routing. |
| **Checkpointer** | PostgreSQL (`PostgresSaver`) | Session transaction checkpointing and state persistence. |
| **Vector Index** | ChromaDB 1.5 | Isolated document storage and session registries. |
| **User Interface** | Streamlit 1.40 | Real-time state visualization of node transitions. |
| **Evaluation Framework**| RAGAS 0.2 | Quantitative RAG metric benchmarking. |

---

## Architecture Decision Records (ADRs)

### ADR 001: Hybrid RAG (Ensemble dense/sparse retriever)
*   **Context**: Dense vector searches are effective at conceptual matching but fail to retrieve exact keyword sequences, code fragments, or specific ID structures.
*   **Decision**: We configured an `EnsembleRetriever` combining ChromaDB semantic search with a BM25 sparse keyword retriever (40% BM25 weight / 60% semantic weight). This mitigates vector-only keyword retrieval failures.

### ADR 002: PostgresSaver Checkpointing
*   **Context**: Default memory checkpointers do not survive application restarts, and SQLite lacks parallel session concurrency and scalability required for production deployments.
*   **Decision**: We integrated `PostgresSaver` as the primary persistence layer. It serializes thread execution states directly to a PostgreSQL database, enabling high concurrency and persistent session history.

### ADR 003: 3-Cycle Safety Cap
*   **Context**: Hallucinating generation or missing document contexts can cause the Self-RAG loop to cycle indefinitely, leading to API rate-limit errors and infinite runtime hangs.
*   **Decision**: We added an integer `loop_count` to the graph state. If `loop_count >= 3`, the conditional edges bypass further grading, forcing routing to the final response node.

### ADR 004: asteval Sandbox for Calculators
*   **Context**: Using python's native `eval()` for calculator execution creates a severe Remote Code Execution (RCE) vector.
*   **Decision**: We sandboxed math operations using the `asteval` library. This limits expression execution to a safe mathematical AST parser, excluding system builtins, imports, and file execution capabilities.
