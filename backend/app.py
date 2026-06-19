import os
import certifi
from .config import settings

# Set SSL Certificate paths for Windows environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Import modules from our package
from .database import get_checkpointer, retrieve_all_threads, register_thread
from .tools import tools as basic_tools, web_search
from .rag import rag_tool, ingest_pdf, thread_has_document, thread_document_metadata, _get_retriever
from .mcp import load_mcp_tools, submit_async_task

# Import CRAG & Self-RAG routines
from .crag import grade_documents_node, rewrite_query_node, web_search_node, decide_to_generate
from .srag import generate_node, grade_generation_decision

# 1. Initialize LLM using centralized config settings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=settings.api_key
)

# 2. Gather all tools
mcp_tools = load_mcp_tools()
all_tools = [*basic_tools, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(all_tools)

# 3. State Schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    generation: str
    query: str
    web_search_needed: bool
    loop_count: int

# 4. Retrieval Integration Node
def retrieve_node(state: ChatState, config=None):
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    query = state.get("query") or ""
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"documents": [], "query": query, "web_search_needed": True}
    docs = retriever.invoke(query)
    return {"documents": docs, "query": query, "web_search_needed": False}

# 5. Wrapper node for final agent delivery
def deliver_response_node(state: ChatState):
    generation = state.get("generation", "")
    return {"messages": [AIMessage(content=generation)]}

# 6. Legacy Chat Node (Acts as dispatcher)
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # If document RAG is active for this thread, route to RAG sub-workflow
    if thread_has_document(thread_id):
        return {"query": "", "documents": [], "generation": "", "loop_count": 0}

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. You can use web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

# Helper to check if RAG path is needed
def route_chat_start(state: ChatState, config=None) -> Literal["retrieve", "chat_node"]:
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    if thread_has_document(thread_id):
        return "retrieve"
    return "chat_node"

# 7. Tool Node
tool_node = ToolNode(all_tools)

# 8. Graph Compilation
checkpointer = get_checkpointer()
graph = StateGraph(ChatState)

# Add Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade_documents", grade_documents_node)
graph.add_node("rewrite_query", rewrite_query_node)
graph.add_node("web_search", web_search_node)
graph.add_node("generate", generate_node)
graph.add_node("deliver_response", deliver_response_node)

# Add Edges
graph.add_conditional_edges(START, route_chat_start, {
    "retrieve": "retrieve",
    "chat_node": "chat_node"
})

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# CRAG & Self-RAG Path
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "rewrite_query",
        "generate": "generate"
    }
)
graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "generate")

graph.add_conditional_edges(
    "generate",
    grade_generation_decision,
    {
        "hallucination": "generate",
        "not_useful": "rewrite_query",
        "useful": "deliver_response"
    }
)
graph.add_edge("deliver_response", END)

chatbot = graph.compile(checkpointer=checkpointer)

