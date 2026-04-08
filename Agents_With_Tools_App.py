from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
import os


if "ALPHAVANTAGE_API_KEY" in st.secrets:
    os.environ["ALPHAVANTAGE_API_KEY"] = st.secrets["ALPHAVANTAGE_API_KEY"]

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    
if "STOCK_API_KEY" in st.secrets:
    os.environ["STOCK_API_KEY"] = st.secrets["STOCK_API_KEY"]
    
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

from Agent_With_Tools import build_workflow

workflow = build_workflow(None)

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Agent",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark terminal aesthetic
# ---------------------------------------------------------------------------

TOOL_ICONS: dict[str, str] = {
    "current_weather": "🌤",
    "get_stock_price": "📈",
    "date_time":       "🕐",
    "internet_search": "🔍",
    "generator":       "📄",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root theme ── */
    :root {
        --bg:        #0d1117;
        --surface:   #161b22;
        --surface2:  #1c2431;
        --border:    #30363d;
        --border2:   #21262d;
        --text:      #e6edf3;
        --muted:     #8b949e;
        --cyan:      #2dd4bf;
        --amber:     #fbbf24;
        --green:     #34d399;
        --red:       #f87171;
        --font-ui:   'Outfit', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* ── App shell ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text);
        font-family: var(--font-ui);
    }
    [data-testid="stHeader"]     { background: transparent !important; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 2rem !important; max-width: 780px !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { font-family: var(--font-ui) !important; }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

    /* ── Sidebar file uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        font-family: var(--font-ui) !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--cyan) !important;
        color: var(--cyan) !important;
        background: #2dd4bf0a !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: var(--surface) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
        font-family: var(--font-ui) !important;
        animation: fadeSlide 0.2s ease forwards;
    }
    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* User bubble — slightly warmer tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
    }

    /* Avatar icons */
    [data-testid="chatAvatarIcon-assistant"] > div {
        background: #2dd4bf18 !important;
        border: 1px solid var(--cyan) !important;
        color: var(--cyan) !important;
    }
    [data-testid="chatAvatarIcon-user"] > div {
        background: #fbbf2418 !important;
        border: 1px solid var(--amber) !important;
        color: var(--amber) !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {
        background: var(--surface) !important;
        border-top: 1px solid var(--border) !important;
        padding: 0.75rem 1rem !important;
        border-radius: 0 0 12px 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: var(--font-ui) !important;
        font-size: 0.92rem !important;
        caret-color: var(--cyan) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 2px #2dd4bf18 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }

    /* Send button */
    [data-testid="stChatInputSubmitButton"] button {
        background: var(--cyan) !important;
        border: none !important;
        border-radius: 6px !important;
    }
    [data-testid="stChatInputSubmitButton"] button:hover {
        background: #20b2a0 !important;
    }
    [data-testid="stChatInputSubmitButton"] svg { fill: #0d1117 !important; }

    /* ── Markdown inside chat ── */
    [data-testid="stChatMessage"] p {
        color: var(--text) !important;
        line-height: 1.75 !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stChatMessage"] code {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--cyan) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
        border-radius: 4px !important;
        padding: 0.1em 0.35em !important;
    }
    [data-testid="stChatMessage"] pre {
        background: #010409 !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    [data-testid="stChatMessage"] pre code {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }

    /* ── Alert / info boxes ── */
    [data-testid="stAlert"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: var(--font-ui) !important;
        font-size: 0.84rem !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div { border-top-color: var(--cyan) !important; }

    /* ── Custom scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

    /* ── Tool pill ── */
    .tool-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #2dd4bf0e;
        border: 1px solid #2dd4bf35;
        color: var(--cyan);
        font-family: var(--font-mono);
        font-size: 0.71rem;
        padding: 2px 8px 2px 6px;
        border-radius: 20px;
        margin-right: 4px;
        animation: pillIn 0.2s ease;
    }
    @keyframes pillIn {
        from { opacity: 0; transform: scale(0.88); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* ── Blinking cursor shown during streaming ── */
    .stream-cursor {
        display: inline-block;
        width: 7px;
        height: 14px;
        background: var(--cyan);
        margin-left: 1px;
        vertical-align: text-bottom;
        border-radius: 1px;
        animation: blink 0.75s step-end infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* ── Mode badge ── */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-bottom: 0.75rem;
    }
    .mode-agent    { background:#2dd4bf0e; border:1px solid #2dd4bf35; color:var(--cyan);  }
    .mode-document { background:#fbbf240e; border:1px solid #fbbf2435; color:var(--amber); }

    /* ── Sidebar stat row ── */
    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
        color: var(--muted);
        padding: 5px 0;
        border-bottom: 1px solid var(--border2);
    }
    .stat-value { color: var(--text); font-weight: 500; }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.63rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 1.25rem 0 0.5rem;
    }

    /* ── Typography ── */
    h1 {
        font-family: var(--font-ui) !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper — HTML builders
# ---------------------------------------------------------------------------

def _tool_pills_html(tool_names: list[str]) -> str:
    pills = "".join(
        f'<span class="tool-pill">{TOOL_ICONS.get(n, "⚙")} {n}</span>'
        for n in tool_names
    )
    return (
        f'<div style="margin-bottom:8px">'
        f'<span style="font-size:0.68rem;color:var(--muted);margin-right:6px">using</span>'
        f'{pills}</div>'
    )


def _mode_badge_html(mode: str) -> str:
    cls   = "mode-document" if mode == "document" else "mode-agent"
    icon  = "📄" if mode == "document" else "⚡"
    label = "Document mode" if mode == "document" else "Agent mode"
    return f'<div class="mode-badge {cls}">{icon}&nbsp;{label}</div>'


# ---------------------------------------------------------------------------
# Core — streaming runner
# ---------------------------------------------------------------------------

def stream_and_render(
    workflow,
    initial_state: dict,
    config: dict,
) -> tuple[str, list[str]]:
    """
    Stream the LangGraph workflow message-by-message.

    Renders tool pills as each tool completes, then streams AI text
    token-by-token with an animated cursor. Returns the full response
    text and a list of tool names that were invoked.
    """
    tool_calls_seen: list[str] = []
    full_response = ""

    tool_placeholder = st.empty()
    text_placeholder = st.empty()

    for chunk, metadata in workflow.stream(initial_state, config, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")

        # ── Tool result chunk ──────────────────────────────────────────────
        if node == "tools":
            name = getattr(chunk, "name", None)
            if name and name not in tool_calls_seen:
                tool_calls_seen.append(name)
                tool_placeholder.markdown(
                    _tool_pills_html(tool_calls_seen),
                    unsafe_allow_html=True,
                )

        # ── AI text chunk ──────────────────────────────────────────────────
        elif node == "chat_node":
            content = ""
            if isinstance(chunk.content, str):
                content = chunk.content
            elif isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")

            if content:
                full_response += content
                text_placeholder.markdown(
                    full_response + '<span class="stream-cursor"></span>',
                    unsafe_allow_html=True,
                )

    # Final render — remove the cursor
    text_placeholder.markdown(full_response)
    if not tool_calls_seen:
        tool_placeholder.empty()   # no tool pills if no tools were used

    return full_response, tool_calls_seen


# ---------------------------------------------------------------------------
# Upload directory & session state
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)


def _init(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


_init("thread_id",      str(uuid.uuid4()))
_init("mode",           "normal")
_init("file_path",      "")
_init("file_name",      "")
_init("file_hash",      "")
_init("document_ready", False)
_init("chat_history",   [])   # list[dict]: role, content, tools_used, ts

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<p style="font-size:1.25rem;font-weight:600;color:var(--text);margin:0 0 2px">⚡ AI Agent</p>'
        '<p style="font-size:0.72rem;color:var(--muted);margin:0 0 1rem">powered by LangGraph</p>',
        unsafe_allow_html=True,
    )

    st.markdown(_mode_badge_html(st.session_state.mode), unsafe_allow_html=True)

    # ── Session stats ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)

    msg_count  = len(st.session_state.chat_history)
    user_turns = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
    all_tools: set[str] = set()
    for m in st.session_state.chat_history:
        all_tools.update(m.get("tools_used", []))

    tools_str = ", ".join(all_tools) if all_tools else "—"
    st.markdown(
        f'<div class="stat-row"><span>Messages</span><span class="stat-value">{msg_count}</span></div>'
        f'<div class="stat-row"><span>Your turns</span><span class="stat-value">{user_turns}</span></div>'
        f'<div class="stat-row" style="border:none"><span>Tools used</span>'
        f'<span class="stat-value" style="font-size:0.68rem">{tools_str}</span></div>',
        unsafe_allow_html=True,
    )

    # ── Document upload ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if st.session_state.document_ready:
        st.markdown(
            f'<div style="font-size:0.74rem;color:var(--green);padding:4px 0 8px">'
            f'✓ &nbsp;{st.session_state.file_name}</div>',
            unsafe_allow_html=True,
        )
        if st.button("✕  Clear document", use_container_width=True):
            st.session_state.update(
                document_ready=False, file_path="", file_name="",
                file_hash="", mode="normal",
                workflow=build_workflow(None), chat_history=[],
            )
            st.rerun()

    # ── Debug ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Debug</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.65rem;color:var(--muted);word-break:break-all">'
        f'thread · {st.session_state.thread_id[:16]}…</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_uploaded_file(file_obj) -> None:
    """Save the file, build the FAISS vector DB, and activate document mode."""
    file_bytes = file_obj.getbuffer()
    file_hash  = hashlib.md5(file_bytes).hexdigest()

    # Same file already indexed — just ensure mode is correct.
    if st.session_state.file_hash == file_hash and st.session_state.document_ready:
        st.session_state.mode = "document"
        return

    file_name  = Path(file_obj.name).name
    save_path  = UPLOAD_DIR / f"{file_hash}_{file_name}"

    with st.spinner(f"Indexing '{file_name}'…"):
        with open(save_path, "wb") as fh:
            fh.write(file_bytes)
        st.session_state.update(
            file_path      = str(save_path),
            file_name      = file_name,
            file_hash      = file_hash,
            workflow       = build_workflow(str(save_path)),
            document_ready = True,
            mode           = "document",
            chat_history   = [],
        )

    st.toast(f"'{file_name}' indexed successfully", icon="📄")


# Handle upload / removal transitions.
if uploaded_file is not None:
    process_uploaded_file(uploaded_file)
else:
    if st.session_state.document_ready:
        # File was removed from the uploader — exit document mode.
        st.session_state.update(
            document_ready = False,
            mode           = "normal",
            workflow       = build_workflow(None),
            chat_history   = [],
        )
    elif st.session_state.mode != "normal":
        st.session_state.mode     = "normal"
        st.session_state.workflow = build_workflow(None)


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 style="margin-bottom:2px">AI Agent '
    '<span style="color:var(--cyan)">with Tools</span></h1>',
    unsafe_allow_html=True,
)

if st.session_state.mode == "document":
    st.markdown(
        f'<p style="font-size:0.83rem;color:var(--amber);margin:0 0 1.25rem">'
        f'📄 &nbsp;Answering from <strong>{st.session_state.file_name}</strong></p>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<p style="font-size:0.83rem;color:var(--muted);margin:0 0 1.25rem">'
        '⚡ &nbsp;Web search &nbsp;·&nbsp; weather &nbsp;·&nbsp; '
        'stocks &nbsp;·&nbsp; date / time</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<hr style="border:none;border-top:1px solid var(--border2);margin:0 0 1rem"/>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        tools_used = message.get("tools_used", [])
        if tools_used and message["role"] == "assistant":
            st.markdown(_tool_pills_html(tools_used), unsafe_allow_html=True)
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input & streaming response
# ---------------------------------------------------------------------------

placeholder_text = (
    "Ask something about your document…"
    if st.session_state.mode == "document"
    else "Ask me anything — search, weather, stocks, time…"
)

user_input = st.chat_input(placeholder_text)

if user_input:
    # Render & store the user turn.
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input, "tools_used": [], "ts": time.time()}
    )

    # Stream the assistant turn.
    with st.chat_message("assistant"):
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        reply, tools_used = stream_and_render(
            st.session_state.workflow, initial_state, config
        )

    # Persist assistant turn (including which tools were called).
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply, "tools_used": tools_used, "ts": time.time()}
    )
