import sys
import os
import uuid
import queue
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Adjust Python path to load backend package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.app import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    ingest_pdf,
    thread_document_metadata,
)
# Import register_thread directly from its source module
from backend.database import register_thread

# Set page config
st.set_page_config(page_title="StateFlow Agentic Assistant", page_icon="🤖", layout="wide")

# =========================== Utilities ===========================
def extract_text_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    extracted.append(item["text"])
                elif "text_content" in item:
                    extracted.append(item["text_content"])
            elif isinstance(item, str):
                extracted.append(item)
        return "".join(extracted)
    elif isinstance(content, dict):
        return content.get("text", "")
    return str(content)

def generate_thread_id():
    username = st.session_state.get("username")
    prefix = f"{username}_" if username else ""
    return f"{prefix}{uuid.uuid4()}"

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    # Register in ChromaDB thread registry
    try:
        register_thread(thread_id, st.session_state.get("username"))
    except Exception:
        pass

def add_thread(thread_id):
    pass

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
        return state.values.get("messages", [])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
        return []

def estimate_thread_tokens_and_cost(thread_id):
    messages = load_conversation(thread_id)
    input_tokens = 0
    output_tokens = 0
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = extract_text_content(msg.content)
            input_tokens += max(1, len(content) // 4)
        elif isinstance(msg, AIMessage):
            content = extract_text_content(msg.content)
            output_tokens += max(1, len(content) // 4)
            
    # Gemini 2.5 Flash pricing: Input ($0.075 / 1M tokens), Output ($0.30 / 1M tokens)
    cost = (input_tokens * (0.075 / 1_000_000)) + (output_tokens * (0.30 / 1_000_000))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": cost
    }

def get_thread_metrics(thread_id):
    thread_key_str = str(thread_id)
    if "token_tracker" not in st.session_state:
        st.session_state["token_tracker"] = {}
    if thread_key_str not in st.session_state["token_tracker"]:
        st.session_state["token_tracker"][thread_key_str] = estimate_thread_tokens_and_cost(thread_id)
    return st.session_state["token_tracker"][thread_key_str]

def update_token_metrics(thread_id: str, prompt_text: str, response_text: str):
    thread_key_str = str(thread_id)
    if "token_tracker" not in st.session_state:
        st.session_state["token_tracker"] = {}
    tracker = st.session_state["token_tracker"].setdefault(thread_key_str, {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0
    })
    
    in_t = max(1, len(prompt_text) // 4)
    out_t = max(1, len(response_text) // 4)
    
    cost = (in_t * (0.075 / 1_000_000)) + (out_t * (0.30 / 1_000_000))
    
    tracker["input_tokens"] += in_t
    tracker["output_tokens"] += out_t
    tracker["total_cost"] += cost

def get_thread_title(thread_id):
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            text = extract_text_content(msg.content).strip()
            if text:
                return text[:20] + "..." if len(text) > 20 else text
    return "New Chat"

# ======================= Session & Authentication ===================
if "username" not in st.session_state:
    st.session_state["username"] = "default_user"

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state or not st.session_state["thread_id"]:
    initial_thread_id = generate_thread_id()
    st.session_state["thread_id"] = initial_thread_id
    # Register this initial thread in ChromaDB
    try:
        register_thread(initial_thread_id, st.session_state.get("username"))
    except Exception:
        pass

# Dynamically load active user's threads on every run
try:
    threads = retrieve_all_threads(st.session_state["username"])[::-1]
except Exception:
    threads = []

# Helper to register thread in active list
if st.session_state["thread_id"] not in threads:
    threads.insert(0, st.session_state["thread_id"])

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("🤖 Chatbot Controls")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# Conversations History List
st.sidebar.subheader("💬 Past Conversations")
if not threads:
    st.sidebar.write("No conversations found.")
else:
    for t_id in threads:
        title = get_thread_title(t_id)
        btn_label = f"💬 {title}"
        if st.sidebar.button(btn_label, key=f"side-thread-{t_id}", use_container_width=True):
            selected_thread = t_id

st.sidebar.divider()

st.sidebar.subheader("📊 Token Economics")
metrics = get_thread_metrics(st.session_state["thread_id"])
st.sidebar.metric("Total Tokens", f"{metrics['input_tokens'] + metrics['output_tokens']:,}")
st.sidebar.metric("Est. Cost", f"${metrics['total_cost']:.6f}")
st.sidebar.caption("Pricing: Gemini 2.5 Flash ($0.075/1M input, $0.30/1M output)")

# ============================ Premium UI/UX Style Injector ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global style resets */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Outfit', sans-serif !important;
        background: radial-gradient(circle at 80% 20%, #15161e 0%, #0c0d12 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* Header/Title alignment styling */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.05rem !important;
    }

    /* Style the Sidebar beautifully */
    [data-testid="stSidebar"] {
        background-color: #0f1016 !important;
        border-right: 1px solid #1f2230 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
    }
    /* Style all sidebar buttons to look premium */
    [data-testid="stSidebar"] button {
        background-color: #171822 !important;
        border: 1px solid #252838 !important;
        color: #d1d5db !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #262938 !important;
        border-color: #7c3aed !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.15) !important;
    }

    /* Custom Chat Message Bubbles (ChatGPT theme) */
    div[data-testid="stChatMessage"] {
        background-color: rgba(23, 24, 34, 0.65) !important;
        border: 1px solid #1f2230 !important;
        border-radius: 20px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        backdrop-filter: blur(8px) !important;
        transition: transform 0.2s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-1px) !important;
    }
    
    /* Style the file uploader wrapper to be a small circle only */
    div[data-testid="stFileUploader"] {
        border: none !important;
        background: transparent !important;
        width: 45px !important;
        height: 45px !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    /* Hide dropzone container borders, background and scrollbars */
    div[data-testid="stFileUploader"] section {
        padding: 0px !important;
        border: none !important;
        background: transparent !important;
        min-height: unset !important;
        width: 100% !important;
        height: 100% !important;
    }
    div[data-testid="stFileUploader"] section > div {
        display: none !important;
    }
    /* Re-style the browse/upload button as a transparent plus icon */
    div[data-testid="stFileUploader"] button {
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: unset !important;
        padding: 0px !important;
        margin: 0px !important;
        font-size: 0px !important; /* Hide 'Upload' text */
        color: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        cursor: pointer !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        position: relative !important;
    }
    div[data-testid="stFileUploader"] button svg {
        display: none !important; /* Hide original upload icon */
    }
    /* Insert plus symbol and center it absolutely */
    div[data-testid="stFileUploader"] button::after {
        content: "+" !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        font-size: 24px !important;
        font-weight: 300 !important;
        color: #b4b4b4 !important;
        line-height: 1 !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Style the main chat input bar */
    div[data-testid="stChatInput"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #f3f4f6 !important;
        font-size: 16px !important;
    }
    
    /* Style status execution indicator boxes */
    div[data-testid="stStatusWidget"] {
        border-radius: 12px !important;
        border: 1px solid #252838 !important;
        background-color: #0f1016 !important;
    }
    
    /* Pin bottom chat bar columns container to the bottom and style it as a single unified ChatGPT pill */
    div[data-testid="stHorizontalBlock"] {
        position: fixed !important;
        bottom: 30px !important;
        width: 60% !important;
        max-width: 800px !important;
        left: 55% !important; /* Offset slightly right to account for sidebar */
        transform: translateX(-50%) !important;
        z-index: 1000 !important;
        background-color: #212121 !important; /* ChatGPT pill background color */
        border: 1px solid #303030 !important;
        border-radius: 28px !important;
        padding: 6px 14px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    /* Adjust when sidebar is collapsed/mobile screen */
    @media (max-width: 991px) {
        div[data-testid="stHorizontalBlock"] {
            left: 50% !important;
            width: 90% !important;
        }
    }
    
    /* Remove padding and margin from columns to merge them seamlessly */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        padding: 0px !important;
        margin: 0px !important;
        min-width: unset !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Pin the active PDF document badge directly above the bottom input bar */
    .active-pdf-badge {
        position: fixed !important;
        bottom: 98px !important;
        left: 55% !important;
        transform: translateX(-50%) !important;
        z-index: 1000 !important;
    }
    @media (max-width: 991px) {
        .active-pdf-badge {
            left: 50% !important;
        }
    }
    
    /* Hide default uploaded file status list inside the bottom bar */
    div[data-testid="stFileUploader"] > section + div {
        display: none !important;
    }
    
    /* Pad bottom of scroll view so messages are not hidden under the bottom bar */
    div[data-testid="stAppViewBlockContainer"] {
        padding-bottom: 220px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================ Main Layout ========================
st.markdown("""
<div style="margin-top: -30px; margin-bottom: 25px;">
    <h1 style="
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa 0%, #3b82f6 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -0.05rem;
    ">
        StateFlow
    </h1>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">
        Equipped with Web Search, Calculator, Stock Tracker, and PDF Document RAG.
    </p>
</div>
""", unsafe_allow_html=True)

# Create a container specifically for the chat history (rendering messages above the input bar)
chat_container = st.container()

# Render existing chat history inside the chat container
with chat_container:
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Bottom controls area (Inline uploader and chat input aligned side-by-side like ChatGPT)
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.markdown(f"""
    <div class="active-pdf-badge" style="
        background-color: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.25);
        padding: 10px 16px;
        border-radius: 16px;
        font-size: 0.95rem;
        color: #c084fc;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.03);
    ">
        <span style="font-size: 1.15rem;">📄</span> 
        <span>Active PDF: <strong>{latest_doc.get('filename')}</strong> ({latest_doc.get('chunks')} chunks, {latest_doc.get('documents')} pages)</span>
    </div>
    """, unsafe_allow_html=True)

# Use columns to position the uploader button directly next to the chat input (narrow upload column)
col_upload, col_chat = st.columns([1, 16], vertical_alignment="bottom")

with col_upload:
    uploaded_pdf = st.file_uploader(
        "PDF", 
        type=["pdf"], 
        label_visibility="collapsed",
        help="Attach PDF to chat",
        key="chat_pdf_uploader"
    )

with col_chat:
    user_input = st.chat_input("Message Chatbot or query document...")

if uploaded_pdf:
    if uploaded_pdf.name not in thread_docs:
        with st.status("🔄 Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed successfully", state="complete", expanded=False)
            st.rerun()

if user_input:
    # Append & display user message INSIDE the chat container (above bottom bar)
    with chat_container:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check conversation billing budget limit ($2.00)
        metrics = get_thread_metrics(thread_key)
        if metrics["total_cost"] >= 2.00:
            st.error("⚠️ Billing limit exceeded ($2.00 maximum per conversation). Please start a new chat thread to continue.")
            st.stop()

        # Invocation Config with LangSmith tracing metadata and tags
        CONFIG = {
            "configurable": {"thread_id": thread_key},
            "metadata": {
                "thread_id": thread_key,
                "user_action": "chat_turn",
            },
            "tags": [
                f"thread_id:{thread_key}",
                "user_action:chat_turn",
            ],
            "run_name": "chat_turn",
        }

        # Assistant Response Generation
        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def stream_response():
                # Run stream generator
                try:
                    for message_chunk, _ in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Tool Executing: `{tool_name}` ...", expanded=True
                                )
                            else:
                                status_holder["box"].update(
                                    label=f"🔧 Tool Executing: `{tool_name}` ...",
                                    state="running",
                                    expanded=True,
                                )

                        if isinstance(message_chunk, AIMessage):
                            yield message_chunk.content
                except Exception as e:
                    yield f"⚠️ An error occurred: {str(e)}"

            ai_message = st.write_stream(stream_response())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished execution", state="complete", expanded=False
                )

        # Save to session history
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

        # Update token tracker metrics and refresh UI
        update_token_metrics(thread_key, user_input, ai_message)
        st.rerun()

        # Show document metadata if active
        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"Active Document: **{doc_meta.get('filename')}** "
                f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
            )

# Switch Conversation Thread
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            temp_messages.append({"role": "assistant", "content": msg.content})
            
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
