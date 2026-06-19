import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages
from .config import settings

# LLM Grader Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=settings.api_key
)

class SelfRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    generation: str
    query: str
    loop_count: int

def generate_node(state: SelfRAGState, config=None) -> dict:
    """
    Generates response using retrieved documents and query.
    """
    docs = state.get("documents", [])
    query = state.get("query", "")
    loop_count = state.get("loop_count", 0) + 1
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""You are a helpful assistant. Answer the user query using the provided document context.
If the context does not contain relevant details, state that you cannot answer.

Context:
{context}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages, config=config)
    return {"generation": response.content, "loop_count": loop_count}

def check_hallucination_srag(docs: List[Document], generation: str) -> str:
    """
    Checks if response is grounded in documents. Returns 'yes' or 'no'.
    """
    if not docs:
        return "yes"
        
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an expert auditor checking for hallucinations.
Given the following context documents, check if the response is fully grounded in and supported by them.
If there are any unsupported facts, claims, or fabrications, reply 'no'. Otherwise, reply 'yes'.
Answer ONLY with 'yes' or 'no'.

Context:
{context}

Response:
{generation}

Is grounded (yes/no):"""
    try:
        res = llm.invoke(prompt).content.strip().lower()
        return "yes" if "yes" in res else "no"
    except Exception:
        return "yes"

def check_answer_srag(query: str, generation: str) -> str:
    """
    Checks if response answers the question. Returns 'yes' or 'no'.
    """
    prompt = f"""You are an expert assistant evaluator.
Determine if the generated response addresses and fully answers the user's question.
If it is useful and directly answers the question, reply 'yes'. Otherwise, reply 'no'.
Answer ONLY with 'yes' or 'no'.

Question: {query}
Response: {generation}

Does it answer the question (yes/no):"""
    try:
        res = llm.invoke(prompt).content.strip().lower()
        return "yes" if "yes" in res else "no"
    except Exception:
        return "yes"

def grade_generation_decision(state: SelfRAGState) -> str:
    """
    Conditional routing edge for Self-RAG loop checks.
    """
    loop_count = state.get("loop_count", 0)
    if loop_count >= 3:
         return "useful"
         
    docs = state.get("documents", [])
    generation = state.get("generation", "")
    query = state.get("query", "")
    
    # Check 1: Hallucination
    if check_hallucination_srag(docs, generation) == "no":
         return "hallucination"
         
    # Check 2: Answer utility
    if check_answer_srag(query, generation) == "no":
         return "not_useful"
         
    return "useful"
