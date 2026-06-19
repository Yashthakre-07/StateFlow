import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages
from .config import settings
from .tools import web_search

# LLM Grader Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=settings.api_key
)

class CRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    query: str
    web_search_needed: bool

def grade_documents_node(state: CRAGState) -> dict:
    """
    Grades retrieved documents for relevance to the user query.
    If <= 50% of the documents are relevant, sets web_search_needed = True.
    """
    docs = state.get("documents", [])
    query = state.get("query", "")
    
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
                
    if not docs:
        return {"web_search_needed": True, "query": query}
        
    grader_prompt = PromptTemplate(
        template="""You are an expert grading assistant.
Evaluate if the following retrieved document context is relevant to the query.
Answer ONLY with 'yes' or 'no'. Do not explain.

Query: {query}
Context: {context}

Relevant (yes/no):""",
        input_variables=["query", "context"]
    )
    
    relevant_docs = []
    relevant_count = 0
    for doc in docs:
        prompt = grader_prompt.format(query=query, context=doc.page_content)
        try:
            res = llm.invoke(prompt).content.strip().lower()
            is_relevant = "yes" in res
        except Exception:
            is_relevant = True
            
        if is_relevant:
            relevant_docs.append(doc)
            relevant_count += 1
            
    web_search_needed = False
    if len(docs) == 0 or (relevant_count / len(docs)) <= 0.5:
        web_search_needed = True
        
    return {"documents": relevant_docs, "web_search_needed": web_search_needed, "query": query}

def rewrite_query_node(state: CRAGState) -> dict:
    """
    Rephrases the query for optimized web or vector store search.
    """
    query = state.get("query", "")
    rewriter_prompt = PromptTemplate(
        template="""You are an expert query optimizer.
Rephrase the following question to make it better suited for semantic search and web search.
Return ONLY the rephrased query and nothing else.

Question: {query}
Optimized Query:""",
        input_variables=["query"]
    )
    try:
        response = llm.invoke(rewriter_prompt.format(query=query)).content.strip()
        new_query = response if response else query
    except Exception:
        new_query = query
    return {"query": new_query}

def web_search_node(state: CRAGState) -> dict:
    """
    Performs web search and appends results to state documents.
    """
    query = state.get("query", "")
    try:
        search_res = web_search.invoke(query)
        web_doc = Document(page_content=str(search_res), metadata={"source": "duckduckgo_web_search"})
        docs = list(state.get("documents", []))
        docs.append(web_doc)
        return {"documents": docs, "web_search_needed": False}
    except Exception:
        return {"web_search_needed": False}

def decide_to_generate(state: CRAGState) -> str:
    """
    Conditional edge deciding whether to search or generate.
    """
    if state.get("web_search_needed", False):
        return "web_search"
    return "generate"
