# 🎥 AI YouTube Chatbot

An AI-powered chatbot that allows users to ask questions about any YouTube video. The application extracts the video's transcript, stores it in a Chroma vector database, retrieves the most relevant content using Retrieval-Augmented Generation (RAG), and generates accurate answers using the Groq LLM.

## 🚀 Features

* Extracts transcripts from YouTube videos.
* Splits transcripts into semantic chunks.
* Stores embeddings in ChromaDB.
* Retrieves relevant context using MMR search.
* Answers questions using the Groq Llama 3.1 model.
* Uses Retrieval-Augmented Generation (RAG) for accurate responses.
* Returns only information available in the video transcript.

---

## 🛠️ Tech Stack

* Python
* LangChain
* ChromaDB
* Groq (Llama 3.1 8B Instant)
* Hugging Face Sentence Transformers
* YouTube Transcript API
* python-dotenv

---

## 📂 Project Structure

```text
youtube-chatbot/
│
├── chatbot.py
├── retriever.py
├── vector_store.py
├── transcript_loader.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/youtube-chatbot.git
cd youtube-chatbot
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Project

```bash
python chatbot.py
```

Example:

```text
Enter Video ID: dQw4w9WgXcQ

Ask a Question:
What is the main topic of the video?
```

---

## 🧠 How It Works

1. User enters a YouTube video ID.
2. Transcript is extracted from the video.
3. Transcript is split into chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in ChromaDB.
6. Relevant chunks are retrieved using MMR search.
7. Groq Llama 3.1 generates the final answer based only on the retrieved context.

---

## 📦 Dependencies

* langchain
* langchain-core
* langchain-community
* langchain-groq
* langchain-huggingface
* langchain-text-splitters
* langchain-chroma
* chromadb
* sentence-transformers
* transformers
* torch
* youtube-transcript-api
* python-dotenv

---

## 📌 Future Improvements

* Streamlit web interface
* Chat history support
* Multi-video knowledge base
* PDF and document support
* Source citations in responses
* Multi-language transcript support

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository, create a feature branch, and submit a pull request.

---

## 📄 License

This project is intended for learning and educational purposes. You may modify and use it according to your requirements.

---

