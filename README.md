# Agentic-AI : MedAI Enterprise — Clinical RAG Healthcare Assistant
# 🏥 MedAI Enterprise — Clinical RAG Healthcare Assistant

An enterprise-grade AI healthcare chatbot powered by Retrieval-Augmented Generation (RAG), designed to provide accurate, context-aware medical responses from clinical documents.

This system ingests medical PDFs, builds semantic vector memory, and enables natural language querying through a conversational interface using Large Language Models.

---

## 🚀 Key Features

- 📄 Medical document ingestion from PDFs
- 🧠 Semantic search using vector embeddings
- 🔎 Context-aware answers grounded in source data
- 💬 Conversational chat interface
- ⚡ Ultra-fast inference via Groq LLM API
- 🏗️ Enterprise-ready RAG architecture
- 🔐 Environment-based secure configuration
- 📚 Source citation support
- 🖥️ Streamlit web application

---

## 🧠 Architecture Overview
PDF Documents
↓
Document Loader (LangChain)
↓
Text Chunking
↓
Embeddings (Sentence Transformers)
↓
FAISS Vector Database
↓
Retriever
↓
LLM (Groq / Hugging Face)
↓
Conversational AI Response


---

## 🛠️ Tech Stack

### 🤖 AI / Machine Learning
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Semantic Search
- Prompt Engineering
- Vector Similarity Search

### 🧩 Frameworks & Libraries
- LangChain (RAG orchestration)
- Hugging Face Transformers
- Sentence Transformers
- FAISS (Facebook AI Similarity Search)
- Streamlit (Web UI)

### ⚡ LLM Providers
- Groq API (Llama models — ultra-fast inference)
- Hugging Face Inference API

### 🗂️ Data Processing
- PyPDFLoader
- DirectoryLoader
- Recursive Text Splitting

### 💻 Backend & Dev Tools
- Python 3.11
- python-dotenv (secure config)
- Virtual Environments (venv)
- Git & GitHub

---

## 📌 System Components

### 1️⃣ Document Ingestion Pipeline

- Loads medical PDFs from local directory
- Extracts text content
- Splits into optimized chunks
- Generates embeddings
- Stores vectors in FAISS database

### 2️⃣ Vector Database

- High-performance similarity search
- Persistent storage
- Enables semantic retrieval

### 3️⃣ Retrieval Engine

- Retrieves relevant context based on user query
- Uses top-k similarity search
- Ensures grounded responses

### 4️⃣ LLM Integration

Supports multiple providers:

- Groq (recommended — fast + free tier)
- Hugging Face Endpoint
- Chat-optimized models

### 5️⃣ Conversational Interface

- Chat-style interaction
- Maintains session history
- Displays source documents

---

## 🏥 Use Cases

- Clinical knowledge assistants
- Hospital internal search tools
- Medical research support
- Healthcare document Q&A
- Patient education systems
- Pharmaceutical knowledge bases

---

## ⚠️ Disclaimer

This system is an AI assistant and not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📂 Project Structure

.
├── pdf_files/ # Source medical documents
├── vectorestore/
│ └── db_faiss/ # FAISS vector database
├── create_memory_with_llm.py # Document ingestion pipeline
├── connect_memory_for_llm.py # CLI-based RAG querying
├── medibot.py # Streamlit chatbot application
├── .env # Environment variables
├── requirements.txt
└── medibot.mp4
├── .gitignore
└── README.md


---

## 🔑 Environment Variables

Create a `.env` file:
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key


---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abhiram1517/Agentic-AI.git
cd Agentic-AI

Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🧾 Step 1 — Build Vector Database
python create_memory_for_llm.py
This will:
Process PDFs Generate embeddings Store vectors in FAISS

💻 Step 2 — Run CLI RAG Query
python connect_memory_with_llm.py

🌐 Step 3 — Launch Web App
streamlit run medibot.py

⚡ Example Queries

"What are canker sores?"

"How to cure cancer?"

"What is cancer?"



🧩 Key Engineering Highlights

Production-ready modular architecture

Pluggable LLM providers

Efficient vector retrieval pipeline

Secure configuration management

Scalable for enterprise deployment

Designed for high-accuracy domain-specific AI

🎯 Why This Project Matters

This system demonstrates real-world implementation of:

✔ Generative AI in healthcare
✔ Enterprise AI architecture
✔ Retrieval-Augmented Generation
✔ LLM integration
✔ End-to-end ML system design

👤 Author
Abhi Ram Bandi

AI Engineer | Data Engineer | Generative AI Enthusiast

⭐ If you find this project useful, consider giving it a star!


---

## 🔥 Why This README Will Impress Recruiters

It signals:

✅ Real production AI skills  
✅ Enterprise architecture thinking  
✅ End-to-end system ownership  
✅ Modern GenAI stack knowledge  
✅ Healthcare domain applicability  
✅ Strong documentation skills  

---

If you want, I can also give you:

🔥 Resume-ready project description  
🔥 LinkedIn project showcase text  
🔥 Interview talking points  
🔥 How to turn this into a SaaS product  
🔥 How to deploy on AWS/Azure  

Just say 🚀