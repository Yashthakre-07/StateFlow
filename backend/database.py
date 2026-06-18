"""
database.py — Thread Persistence & LangGraph Checkpointing

Architecture:
  - ChromaDB (PersistentClient): Primary store for thread registry & session metadata.
    Each user's threads are tracked as documents in the `thread_registry` collection.
    ChromaDB is imported lazily so the module loads safely in all environments.
  - SqliteSaver (LangGraph): Handles LangGraph graph-state checkpointing internally.
    Chroma cannot replace this layer (requires LangGraph's binary serialization format).
"""

import sqlite3
from typing import Optional
from datetime import datetime, timezone

from langgraph.checkpoint.sqlite import SqliteSaver
from .config import settings


# ─────────────────────────────────────────────────
# 1.  ChromaDB — Primary Thread Registry Database
#     (lazy import — graceful fallback if not installed)
# ─────────────────────────────────────────────────

_chroma_client = None
_thread_registry = None


def _get_chroma_registry():
    """Lazily initialize ChromaDB client and thread_registry collection."""
    global _chroma_client, _thread_registry
    if _thread_registry is not None:
        return _thread_registry
    try:
        import chromadb
        if settings.chroma_host:
            _chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        else:
            _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _thread_registry = _chroma_client.get_or_create_collection(
            name="thread_registry",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        _thread_registry = None
    return _thread_registry


def register_thread(thread_id: str, username: Optional[str] = None) -> None:
    """Register a new thread in the ChromaDB thread registry."""
    registry = _get_chroma_registry()
    if registry is None:
        return  # ChromaDB not available — skip silently

    doc_id = str(thread_id)
    try:
        existing = registry.get(ids=[doc_id])
        if existing and existing["ids"]:
            return  # already registered
    except Exception:
        pass

    meta = {
        "username": username or "default_user",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        registry.add(
            ids=[doc_id],
            documents=[doc_id],
            metadatas=[meta],
        )
    except Exception:
        pass


def retrieve_all_threads(username: Optional[str] = None) -> list:
    """
    Return all thread IDs for the given user.
    Queries ChromaDB first; falls back to scanning SqliteSaver checkpoints.
    """
    registry = _get_chroma_registry()
    if registry is not None:
        try:
            results = registry.get(
                where={"username": username or "default_user"} if username else None,
                include=["metadatas"],
            )
            if results and results["ids"]:
                # Sort by created_at descending (newest first)
                paired = list(zip(results["ids"], results["metadatas"]))
                paired.sort(
                    key=lambda x: x[1].get("created_at", ""),
                    reverse=True,
                )
                return [t[0] for t in paired]
        except Exception:
            pass

    # ── Backward-compat fallback: scan SqliteSaver ──
    all_threads: set = set()
    try:
        for checkpoint in checkpointer.list(None):
            t_id = checkpoint.config["configurable"]["thread_id"]
            if username:
                if t_id.startswith(f"{username}_"):
                    all_threads.add(t_id)
            else:
                all_threads.add(t_id)
    except Exception:
        pass
    return list(all_threads)


def get_thread_metadata(thread_id: str) -> dict:
    """Fetch stored metadata for a specific thread from ChromaDB."""
    registry = _get_chroma_registry()
    if registry is None:
        return {}
    try:
        result = registry.get(ids=[str(thread_id)], include=["metadatas"])
        if result and result["metadatas"]:
            return result["metadatas"][0]
    except Exception:
        pass
    return {}


def update_thread_title(thread_id: str, title: str) -> None:
    """Update the human-readable title of a thread stored in ChromaDB."""
    registry = _get_chroma_registry()
    if registry is None:
        return
    try:
        existing = registry.get(ids=[str(thread_id)], include=["metadatas"])
        if existing and existing["metadatas"]:
            meta = existing["metadatas"][0]
            meta["title"] = title[:80]
            registry.update(ids=[str(thread_id)], metadatas=[meta])
    except Exception:
        pass


# ─────────────────────────────────────────────────
# 2.  SqliteSaver — LangGraph State Checkpointer
# ─────────────────────────────────────────────────

checkpointer: Optional[object] = None

# Try PostgresSaver first (production-grade)
if settings.postgres_url:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver.from_conn_string(settings.postgres_url)
    except ImportError:
        pass

# Fallback: local SQLite for LangGraph graph-state snapshots
if checkpointer is None:
    _sqlite_conn = sqlite3.connect(
        database=settings.db_path,
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn=_sqlite_conn)


def get_checkpointer():
    return checkpointer
