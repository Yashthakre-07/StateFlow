import os
import tempfile
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from .config import settings

_persistent_client = None

def _get_chroma_client():
    global _persistent_client
    if _persistent_client is not None:
        return _persistent_client
    import chromadb
    if settings.chroma_host:
        _persistent_client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        _persistent_client = chromadb.PersistentClient(path="./chroma_db")
    return _persistent_client

METADATA_FILE = "./chroma_db/collection_metadata.json"

# Helper functions to load/save thread metadata to disk
def _load_all_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_metadata(thread_id: str, meta: dict):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    all_meta = _load_all_metadata()
    all_meta[str(thread_id)] = meta
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(all_meta, f)
    except Exception:
        pass

# Embeddings helper (supports OpenAI or Google Gemini depending on env keys)
def get_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.api_key)

def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    collection_name = f"thread_{str(thread_id).replace('-', '_')}"
    if thread_has_document(thread_id):
        embeddings = get_embeddings()
        from langchain_chroma import Chroma
        vector_store = Chroma(
            client=_get_chroma_client(),
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        
        chroma_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        try:
            # Retrieve documents from Chroma to build BM25 sparse index dynamically
            chroma_data = vector_store.get()
            if chroma_data and chroma_data.get("documents"):
                from langchain_core.documents import Document
                from langchain_community.retrievers import BM25Retriever
                from langchain.retrievers import EnsembleRetriever
                
                docs = [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in zip(chroma_data["documents"], chroma_data["metadatas"])
                ]
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 4
                
                # Ensemble Retriever (Hybrid search combining Keyword + Semantic)
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, chroma_retriever],
                    weights=[0.4, 0.6]
                )
                return ensemble_retriever
        except Exception:
            # Graceful fallback to pure Chroma if dependencies fail
            pass
            
        return chroma_retriever
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        embeddings = get_embeddings()
        collection_name = f"thread_{str(thread_id).replace('-', '_')}"
        
        # Reset/delete collection if it already exists to index fresh
        try:
            _get_chroma_client().delete_collection(collection_name)
        except Exception:
            pass

        from langchain_chroma import Chroma
        vector_store = Chroma(
            client=_get_chroma_client(),
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        vector_store.add_documents(chunks)

        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _save_metadata(thread_id, meta)

        return meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# Pydantic input validation schema for RAG Search
class RAGInput(BaseModel):
    query: str = Field(..., description="The query to search the PDF for.")
    thread_id: str = Field(..., description="The unique thread ID of the active chat session.")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 200:
            raise ValueError("Query too long (max 200 characters).")
        return v

@tool("rag_tool", args_schema=RAGInput)
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }

def thread_has_document(thread_id: str) -> bool:
    collection_name = f"thread_{str(thread_id).replace('-', '_')}"
    try:
        col = _get_chroma_client().get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False

def thread_document_metadata(thread_id: str) -> dict:
    return _load_all_metadata().get(str(thread_id), {})
