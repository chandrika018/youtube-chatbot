# 🎥 YouTube & Document Chat Assistant

This project now provides a production-ready Streamlit experience for chatting with YouTube transcripts and uploaded documents using retrieval-augmented generation (RAG).

## ✅ What changed

- Replaced the broken graph-based UI with a stable Streamlit app.
- Added modular loaders for YouTube transcripts and PDF/DOCX/TXT files.
- Added FAISS-backed vector stores that persist per source.
- Added a combined retriever for YouTube, documents, or both.
- Added graceful error handling for invalid URLs, unsupported files, empty content, and missing API keys.
- Added regression tests for URL parsing and document validation.

## 🚀 Features

- Paste a YouTube URL and process its transcript.
- Upload multiple PDF, DOCX, or TXT files.
- Choose YouTube, Documents, or Both as the knowledge source.
- Ask questions and receive grounded answers from retrieved context.
- Keep chat history during the session.
- Reset the chat or the processed knowledge state.

## 🧱 Project structure

```text
youtube-chat/
├── app.py
├── chatbot.py
├── docuLoader.py
├── embedding.py
├── loaders/
├── processing/
├── prompts/
├── retrievers/
├── tests/
├── utils/
├── vectorstore/
└── requirements.txt
```

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🔐 Environment variables

Create a .env file with:

```env
GROQ_API_KEY=your_groq_api_key
```

If no API key is present, the app will fall back to a lightweight local response mode so the UI still runs.

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 🧪 Tests

```bash
pytest -q tests/test_pipeline.py
```

## 📝 Notes

- The current implementation uses FAISS for vector storage.
- YouTube transcript fetching relies on the YouTube Transcript API and may fail for private or transcript-disabled videos.
- Document loading supports PDF, DOCX, and TXT files only.