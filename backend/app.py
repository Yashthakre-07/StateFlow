import os
import certifi
from .config import settings

# Set SSL Certificate paths for Windows environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Import modules from our package
from .database import get_checkpointer, retrieve_all_threads, register_thread
from .tools import tools as basic_tools
from .rag import rag_tool, ingest_pdf, thread_has_document, thread_document_metadata
from .mcp import load_mcp_tools, submit_async_task

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

# 4. Chat Node
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

# 5. Tool Node
tool_node = ToolNode(all_tools)

# 6. Graph Compilation
checkpointer = get_checkpointer()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)
