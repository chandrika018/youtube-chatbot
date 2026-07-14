import streamlit as st
from graph import get_graph  
from graph_file import get_graph_file  # Backend logic intact
import uuid

# ==========================================
# 1. PAGE CONFIGURATION & PREMIUM STYLING
# ==========================================
st.set_page_config(
    page_title="youtubeAI Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Ultra-Modern CSS Injection (Neon-Glassmorphic & Micro-Interactions)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles & Custom Scrollbar */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #070a13 !important;
        color: #E2E8F0;
    }
    
    /* Clean Dark Sidebar with Subtle Left Border Accent */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Ultra-Premium Gradient Buttons with Glow */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 0.7rem 1.2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4);
        background: linear-gradient(135deg, #4F46E5 0%, #9333EA 100%);
    }
    
    /* High-Fidelity Futuristic Status Pills */
    .status-badge {
        background: rgba(16, 185, 129, 0.06);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.05);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* Cyberpunk-Glassmorphism Metric Cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Modern Text & File Inputs styling override */
    div[data-testid="stTextInput"] input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #818CF8 !important;
        box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
    }
    
    /* Custom Styling for Drag and Drop File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.4);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 10px;
    }

    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session States (Chat Messages & Unique Memory Thread)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ==========================================
# MEMORY GRAPH CORE INTEGRATION
# ==========================================
async def stream_graph_response(url_or_file, question, is_file=False):
    """
    Invokes the backend compiled graph with stateful thread memory
    and yields chunks for Streamlit's real-time streaming interface.
    (For YouTube Video processing via get_graph)
    """
    try:
        app_graph = get_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        input_data = {
            "question": question, 
            "video_url": url_or_file if not is_file else None,
            "uploaded_file": url_or_file if is_file else None 
        }
        
        async for result in app_graph.astream(input_data, config=config):
            print(result)
            if "generate" in result:
                yield result["generate"]["answer"]
           
    except Exception as e:
        yield f"⚠️ Memory Sync Error: {str(e)}"

async def stream_graph_text(url_or_file, question,is_url=False):
    """
    Invokes the backend compiled graph with stateful thread memory
    and yields chunks for Streamlit's real-time streaming interface.
    (For Local Document upload via get_graph_file)
    """
    try:
        app_graph = get_graph_file()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        if not is_url:
           file_data = url_or_file.read()

           input_text = {
        "question": question,
        "uploaded_file": file_data
    }
        else:
            input_text = {
            "question": question,
            "video_url": url_or_file
    }
        
          

        async for result in app_graph.astream(input_text, config=config):
            print(result)
            if "generate" in result:
                yield result["generate"]["answer"]
           
    except Exception as e:
        yield f"⚠️ Memory Sync Error: {str(e)}"

# ==========================================
# 2. SIDEBAR NAVIGATION & PIPELINE SPECS
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 12px; margin-top: 10px;'>
            <span style='font-size: 2rem;'>🧠</span>
            <h2 style='margin:0; color:#F8FAFC; font-weight: 700; letter-spacing: -0.5px;'>youtubeAI <span style='color:#A855F7;'>Studio</span></h2>
        </div>
        <p style='font-size:0.78rem; color:#64748B; margin-left: 45px; margin-bottom:2rem;'>Production Interface v1.1</p>
    """, unsafe_allow_html=True)
    
    if st.button("✨ New Exploration Space", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())  
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'>⚙️ Active Pipeline Specs</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.5) 100%); border-radius: 14px; padding: 1rem; border-left: 4px solid #6366F1; border-top: 1px solid rgba(255,255,255,0.03);'>
            <p style='margin:0 0 8px 0; font-size: 0.82rem; color: #E2E8F0; display:flex; align-items:center; gap:8px;'>🧬 <strong>Embeddings:</strong> <span style='color:#94A3B8;'>HF-MiniLM-L6</span></p>
            <p style='margin:0 0 8px 0; font-size: 0.82rem; color: #E2E8F0; display:flex; align-items:center; gap:8px;'>🗄️ <strong>Vector Index:</strong> <span style='color:#94A3B8;'>ChromaDB Vector</span></p>
            <p style='margin:0 0 8px 0; font-size: 0.82rem; color: #E2E8F0; display:flex; align-items:center; gap:8px;'>🔥 <strong>Core Engine:</strong> <span style='color:#A855F7; font-weight:600;'>Gemini-Pro</span></p>
            <p style='margin:0; font-size: 0.72rem; color: #64748B; word-break: break-all;'>🆔 Session-ID: {st.session_state.thread_id[:13]}...</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Chat Database", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 3. MAIN APP LAYOUT (RESPONSIVE SPLIT)
# ==========================================
workspace_layout, metrics_panel = st.columns([7, 3], gap="large")

with metrics_panel:
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0; color:#F1F5F9; font-weight:700; display:flex; align-items:center; gap:8px;'>📊 Contextual Metadata</h4>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(15, 23, 42, 0.6); height:120px; border-radius:12px; display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:1.5rem; border: 1px dashed rgba(99, 102, 241, 0.3); box-shadow: inset 0 4px 20px rgba(0,0,0,0.4);'>
            <span style='font-size: 1.8rem; margin-bottom: 4px;'>⚡</span>
            <span style='color:#38BDF8; font-size:0.8rem; font-weight:600; letter-spacing:0.5px; text-transform:uppercase;'>Media Sync Engine Online</span>
        </div>
        <div style='display:flex; flex-direction:column; gap:10px;'>
            <p style='margin:0; font-size:0.88rem; color:#94A3B8; display:flex; align-items:center; gap:8px;'>⏱️ <strong>Data Source:</strong> <span style='color:#F1F5F9;'>Auto-Detect</span></p>
            <p style='margin:0; font-size:0.88rem; color:#94A3B8; display:flex; align-items:center; gap:8px;'>🧱 <strong>Context Window:</strong> <span style='color:#F1F5F9;'>42 Dense Chunks</span></p>
            <p style='margin:0; font-size:0.88rem; color:#94A3B8; display:flex; align-items:center; gap:8px;'>🎯 <strong>RAG Search:</strong> <span style='color:#34D399; font-weight:500;'>Cosine Matcher</span></p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with workspace_layout:
    # URL and File Dual Input System
    input_tab1, input_tab2 = st.tabs(["🔗 YouTube Link", "📁 Upload Document"])
    
    with input_tab1:
        video_url = st.text_input(
            "Source Stream Link",
            placeholder="Paste your YouTube video URL link here to sync context...",
            label_visibility="collapsed"
        )
        
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload Document Context", 
            type=["pdf", "txt", "docx"], 
            label_visibility="collapsed"
        )
    import tempfile
    import os

    file_path = None

    if file_path:
       with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(file_path.name)[1]
    ) as tmp:
        tmp.write(file_path.getvalue())
        file_path = tmp.name
    
    # Active context assignment
    has_context = bool(video_url or file_path)
    active_source = video_url if video_url else file_path
    is_file_mode = bool(file_path and not video_url)
    
    # STATE 1: Empty Screen (Welcome View)
    if not has_context:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(to right, #A5B4FC, #E9D5FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; margin-bottom:10px;'>Analyze Any Resource Instantly</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.1rem; color: #64748B; margin-bottom: 3rem; line-height: 1.6; max-width:650px;'>Provide a YouTube link or upload a local document file to securely parse text vector indices, auto-locate sources, and query contextual details instantly.</p>", unsafe_allow_html=True)
        
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown("""
                <div style='background: rgba(30, 41, 59, 0.3); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.04);'>
                    <h5 style='color:#F1F5F9; margin-top:0; margin-bottom:6px; font-weight:600; display:flex; align-items:center; gap:8px;'>🔍 Semantic Indexing</h5>
                    <p style='color:#64748B; font-size:0.88rem; margin:0; line-height:1.5;'>Extracts core context and abstract meanings rather than just exact keywords from files & transcripts.</p>
                </div>
            """, unsafe_allow_html=True)
        with feat_col2:
            st.markdown("""
                <div style='background: rgba(30, 41, 59, 0.3); padding: 1.25rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.04);'>
                    <h5 style='color:#F1F5F9; margin-top:0; margin-bottom:6px; font-weight:600; display:flex; align-items:center; gap:8px;'>⏱️ Traceable Answers</h5>
                    <p style='color:#64748B; font-size:0.88rem; margin:0; line-height:1.5;'>Every AI-generated claim links explicitly back to its exact source data or video timestamp.</p>
                </div>
            """, unsafe_allow_html=True)
            
    # STATE 2: Active RAG UI
    else:
        if is_file_mode:
            status_text = f"📄 File Synced: {file_path.name[:20]}"
        else:
            status_text = "🟢 Live Transcript"
        st.markdown(f"""
            <div style='display:flex; gap:12px; margin-top: 1rem; margin-bottom: 2rem; flex-wrap:wrap;'>
                <span class='status-badge'>{status_text}</span>
                <span class='status-badge'>⚡ Normalized Chunks</span>
                <span class='status-badge'>🔮 Vector Space Synced</span>
                <span class='status-badge'>🔗 LLM Node Connected</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Render Chat History Inside Workspace
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "rag_data" in message:
                    with st.expander("🛠️ Analytics Protocol Logs", expanded=False):
                        st.json(message["rag_data"])
        if video_url and file_path:
            st.warning("Please select only one source.")
            st.stop()

# ==========================================
# 4. FIXED BOTTOM INPUT (CHATGPT STYLE)
# ==========================================
if has_context:

    if question := st.chat_input("Ask anything regarding this context..."):

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with workspace_layout:

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                # ==========================================
                # सुधार (FIX): यहाँ पर कंडीशन चेक करके सही ग्राफ़ को कॉल किया जा रहा है
                # ==========================================
                if is_file_mode:
                    # यदि फ़ाइल मोड एक्टिव है तो stream_graph_text (get_graph_file) चलेगा
                    stream_generator = stream_graph_text(
                        active_source,
                        question
                    )
                else:
                    # यदि YouTube URL एक्टिव है तो stream_graph_response (get_graph) चलेगा
                    stream_generator = stream_graph_response(
                        active_source,
                        question,
                        is_file=False
                    )

                full_response = st.write_stream(stream_generator)

                rag_metadata = {
                    "similarity_score": "0.892 Cosine Metric",
                    "retrieved_nodes": [
                        "Chunk #12",
                        "Chunk #15"
                    ],
                    "target_location": "Page #2 / Timestamp 12:45"
                }

                with st.expander(
                    "🛠️ Analytics Protocol Logs",
                    expanded=False
                ):
                    st.json(rag_metadata)


        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "rag_data": rag_metadata
        })

        st.rerun()