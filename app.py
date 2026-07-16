import os
import uuid
import time  
import streamlit as st

from agents.chat_agent import ChatAgent
from loaders.documents import load_documents_from_uploads
from loaders.youtube import load_youtube_transcript
from utils.logging import logger
from vectorstore.faiss_store import KnowledgeBase
from langchain_core.documents import Document


st.set_page_config(page_title="YouTube & Document Chat Assistant", page_icon="🧠", layout="wide")


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "youtube_store": None,
        "document_store": None,
        "youtube_source": None,
        "document_source": None,
        "retriever": None,
        "vector_store": None,
        "documents": [],
        "uploaded_file": None,
        "chunks": [],
        "embeddings": None,
        "chat_history": [],
        "graph_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #0d1117;
        --bg-subtle: #161b22;
        --border: #30363d;
        --text: #e6edf3;
        --text-muted: #8b949e;
        --accent: #388bfd;
        --accent-hover: #58a6ff;
        --radius: 6px;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: var(--bg-subtle);
        border-right: 1px solid var(--border);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    .card {
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        box-shadow: none;
    }
    div[data-testid="stFileUploader"] > section {
        border-radius: var(--radius);
        border: 1px dashed var(--border);
        padding: 0.9rem;
        background: var(--bg-subtle);
    }
    .source-box {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.6rem 0.75rem;
        background: var(--bg-subtle);
        margin-top: 0.4rem;
    }
    .source-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .source-value {
        font-size: 0.9rem;
        color: var(--text);
        margin-top: 0.15rem;
        word-break: break-word;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stFileUploader > div {
        background: var(--bg-subtle);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }
    .stButton > button {
        border-radius: var(--radius);
        background: var(--accent);
        color: #ffffff;
        border: 1px solid var(--accent);
        font-weight: 500;
        transition: background 0.1s ease;
    }
    .stButton > button:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
    }
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: var(--radius);
    }
    hr {
        border-color: var(--border);
    }
    /* Ensure Streamlit's own text elements (markdown, captions, labels) render white/light on dark bg */
    p, span, label, li, .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.youtube_store = None
    st.session_state.document_store = None
    st.session_state.youtube_source = None
    st.session_state.document_source = None
    st.session_state.retriever = None
    st.session_state.vector_store = None
    st.session_state.documents = []
    st.session_state.uploaded_file = None
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.chat_history = []
    st.session_state.graph_state = None


def show_status(message: str, kind: str = "info") -> None:
    if kind == "success":
        st.success(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)


with st.sidebar:
    st.markdown(
        """
        <div style='padding: 0.3rem 0 1rem 0;'>
            <h2 style='margin:0; color:#e6edf3; font-size:1.15rem; font-weight:600;'>🧠 YouTube &amp; Document Chat</h2>
            <p style='margin:0.25rem 0 0 0; color:#8b949e; font-size:0.85rem;'>Retrieval-augmented chat assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    st.caption("You can provide a YouTube link and upload documents together.")

    col1, col2 = st.columns(2)
    with col1:
        process_clicked = st.button("Process", use_container_width=True)
    with col2:
        reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        reset_session()
        st.rerun()

    if process_clicked:
        has_youtube = bool(youtube_url and youtube_url.strip())
        has_documents = bool(uploaded_files)

        if not has_youtube and not has_documents:
            show_status("Please provide a YouTube URL or upload at least one document.", kind="warning")
            st.stop()

        try:
            with st.spinner("Processing your sources..."):
                if has_youtube:
                    t0 = time.perf_counter()
                    youtube_payload = load_youtube_transcript(youtube_url)
                    t1 = time.perf_counter()
                    logger.info("Transcript fetch took %.2f sec", t1 - t0)
                    logger.info("Transcript length: %d characters", len(youtube_payload["text"]))

                    youtube_store_dir = os.path.join("vector_store", youtube_payload["video_id"])
                    youtube_store = KnowledgeBase(persist_dir=youtube_store_dir)
                    youtube_store.reset()

                    t2 = time.perf_counter()
                    youtube_store.add_documents(
                        [
                            Document(
                                page_content=youtube_payload["text"],
                                metadata={"source": "youtube", "title": youtube_payload["title"], "url": youtube_payload["url"]},
                            )
                        ],
                        source="youtube",
                    )
                    t3 = time.perf_counter()
                    logger.info("Embedding + FAISS build took %.2f sec", t3 - t2)

                    st.session_state.youtube_store = youtube_store
                    st.session_state.youtube_source = youtube_payload["url"]
                    st.session_state.retriever = None
                    st.session_state.graph_state = None
                    st.session_state.chat_history = []
                    logger.info("Loaded YouTube transcript for %s", youtube_payload["video_id"])

                if has_documents:
                    doc_objects = load_documents_from_uploads(uploaded_files)
                    document_store_dir = os.path.join("vector_store", "documents")
                    document_store = KnowledgeBase(persist_dir=document_store_dir)
                    document_store.reset()
                    document_store.add_documents(doc_objects, source="document")
                    st.session_state.document_store = document_store
                    st.session_state.document_source = [getattr(uploaded_file, "name", "") for uploaded_file in uploaded_files if getattr(uploaded_file, "name", "")]
                    st.session_state.documents = doc_objects
                    st.session_state.uploaded_file = uploaded_files[0] if uploaded_files else None
                    st.session_state.chunks = []
                    st.session_state.embeddings = None
                    st.session_state.retriever = None
                    st.session_state.graph_state = None
                    st.session_state.chat_history = []
                    logger.info("Loaded %d uploaded documents", len(doc_objects))

            show_status("Sources processed successfully.", kind="success")
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Processing failed")
            show_status(f"Processing failed: {exc}", kind="error")
            st.stop()

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <div class='card'>
        <h1 style='margin:0 0 0.35rem 0; font-size:1.5rem; font-weight:600; color:#e6edf3;'>Ask questions from your uploaded content and YouTube videos</h1>
        <p style='margin:0; color:#8b949e; font-size:0.92rem;'>Answers are generated using retrieval-augmented generation with source-aware context.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources", expanded=False):
                    for item in message["sources"]:
                        source_type = item.get("source") or "source"
                        title = item.get("title") or item.get("filename") or source_type
                        detail = item.get("url") or item.get("filename") or ""
                        st.markdown(
                            f"<div class='source-box'><div class='source-label'>{source_type}</div><div class='source-value'>{title}</div>"
                            + (f"<div class='source-value'>{detail}</div>" if detail else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Process a YouTube URL or upload documents to start chatting.")

question = st.chat_input("Ask a question about the processed content")
if question:
    if not st.session_state.get("youtube_store") and not st.session_state.get("document_store"):
        show_status("Process a source before asking questions.", kind="warning")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            agent = ChatAgent(
                youtube_store=st.session_state.get("youtube_store"),
                document_store=st.session_state.get("document_store"),
            )
            result = agent.build_graph().invoke(
                {
                    "question": question,
                    "context": "",
                    "answer": "",
                    "source": "both" if st.session_state.get("youtube_store") and st.session_state.get("document_store") else ("youtube" if st.session_state.get("youtube_store") else "documents"),
                }
            )
            answer = result.get("answer", "I couldn't produce an answer.")
            st.markdown(answer)
            source_payload = result.get("sources", []) or []
            if source_payload:
                with st.expander("Sources", expanded=False):
                    for item in source_payload:
                        source_type = item.get("source") or "source"
                        title = item.get("title") or item.get("filename") or source_type
                        detail = item.get("url") or item.get("filename") or ""
                        st.markdown(
                            f"<div class='source-box'><div class='source-label'>{source_type}</div><div class='source-value'>{title}</div>"
                            + (f"<div class='source-value'>{detail}</div>" if detail else "")
                            + "</div>",
                            unsafe_allow_html=True,
                        )
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_payload})

    st.rerun()

st.markdown(
    "<div style='margin-top: 2.5rem; text-align: center; color: #8b949e; font-size: 0.78rem;'>Built with Streamlit, LangChain, FAISS, and Hugging Face</div>",
    unsafe_allow_html=True,
)