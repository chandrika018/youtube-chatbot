import streamlit as st
from chatbot import ask_question


# Page Config
st.set_page_config(
    page_title="YouTube AI Chatbot",
    page_icon="🧠",
    layout="wide"
)


# Title
st.title("🧠 YouTube AI Chatbot")
st.write(
    "Ask questions about any YouTube video using RAG + Gemini"
)


# Sidebar
with st.sidebar:

    st.header("⚙️ Settings")

    video_id = st.text_input(
    "Enter YouTube Video ID"
    )

    st.divider()

    st.info(
        """
        Pipeline:
        
        ✅ Transcript Extraction
        
        ✅ Text Chunking
        
        ✅ HuggingFace Embeddings
        
        ✅ Chroma Vector DB
        
        ✅ Gemini LLM
        """
    )


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])



# User Input
question = st.chat_input(
    "Ask a question about this video..."
)


if question:


    # Show user message
    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):
        st.write(question)



    # AI Response
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching video and generating answer..."
        ):

            answer = ask_question(
                video_id,
                question
            )


        st.write(answer)


        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )