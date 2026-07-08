# 🎥 YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about any YouTube video using its transcript. The application automatically extracts the transcript, creates embeddings, stores them in a FAISS vector database, retrieves the most relevant context, and generates accurate answers using Meta Llama 3.1 through the Hugging Face Inference API.

---

## 🚀 Features

* 🎥 Accepts a YouTube URL or Video ID
* 📜 Automatically fetches available YouTube transcripts
* 🌍 Automatically detects transcript language
* ✂️ Splits transcripts into semantic chunks
* 🧠 Generates embeddings using Sentence Transformers
* 📚 Stores embeddings in a FAISS vector database
* 🔎 Retrieves the most relevant transcript chunks
* 🤖 Answers questions using Meta Llama 3.1
* 💬 Modern Streamlit chat interface
* ⚡ Fast semantic search with Retrieval-Augmented Generation (RAG)

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### AI Models

* Meta Llama 3.1 8B Instruct
* Sentence Transformers (all-MiniLM-L6-v2)

### Vector Database

* FAISS

### APIs

* Hugging Face Inference API
* YouTube Transcript API

---

## 📂 Project Structure

```text
YouTube-RAG/
│
├── app.py
├── .env
├── requirements.txt
├── README.md
└── assets/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-rag-assistant.git

cd youtube-rag-assistant
```

---

### 2. Create a Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Create a `.env` File

```env
HUGGINGFACE_API_TOKEN=YOUR_HUGGINGFACE_API_KEY
```

---

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. User enters a YouTube URL or Video ID.
2. The transcript is downloaded automatically.
3. The transcript is split into manageable chunks.
4. Sentence Transformer embeddings are created.
5. Embeddings are stored in a FAISS vector database.
6. User asks a question.
7. Relevant transcript chunks are retrieved.
8. Meta Llama 3.1 generates an answer based only on the retrieved context.

---

## 🖥️ Application Workflow

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Text Chunking
      │
      ▼
Sentence Transformer Embeddings
      │
      ▼
FAISS Vector Database
      │
      ▼
Retriever
      │
      ▼
Prompt Template
      │
      ▼
Meta Llama 3.1
      │
      ▼
Answer
```

---

## 📦 Main Libraries

* streamlit
* langchain
* langchain-huggingface
* langchain-community
* langchain-text-splitters
* faiss-cpu
* sentence-transformers
* youtube-transcript-api
* huggingface-hub
* python-dotenv

---

## 🎯 Use Cases

* Educational video assistant
* Lecture question answering
* YouTube content summarization
* Technical tutorial assistant
* Research support
* Podcast knowledge retrieval
* AI-powered learning companion

---

## 📸 Screenshots

Add screenshots of:

* Home Screen
* Knowledge Base Creation
* Chat Interface
* Sample Question & Answer

---

## 🔮 Future Improvements

* Multiple YouTube video support
* PDF document support
* Website URL support
* Persistent vector database
* Conversation memory
* Source citations with timestamps
* Multi-language translation
* Voice input and speech output
* Authentication and user accounts
* Docker deployment
* Cloud deployment

---

## 👨‍💻 Author

**Hammad Younis Abbasi**

BS Computer Science Student

Backend Developer | AI & Agentic AI Enthusiast

---

## ⭐ Support

If you found this project helpful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🛠️ Contribute improvements
* 📝 Share feedback and suggestions

Your support helps improve the project and encourages future development.
