# 🧠 Modular RAG PDF Chatbot with FastAPI, ChromaDB, Langchain
 ![Deskripsi gambar](assets/RAG%20Image.png)

This project is a modular **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and chat with an AI assistant that answers queries based on the document content. It features a microservice architecture with a decoupled **FastAPI backend**, using **ChromaDB** as the vector store, **Groq's LLaMA3 model** as the LLM, and **OpenAi gpt-4o-mini model** as NLP.

---

## 📂 Project Structure

```
FASTAPI-RAG/
│   ├── chroma_store/ ....after run
|   |──modules/
│      ├── load_vectorestore.py
│      ├── rag_setup.py
|   |──uploaded_pdfs/ ....after run
│   ├── config.py
│   └── logger.py
│   └── main.py
└── README.md
```

---

## ✨ Features

- 📄 Upload and parse PDFs
- 🧠 Embed document chunks with HuggingFace embeddings
- 💂️ Store embeddings in ChromaDB
- 💬 Query documents using LLaMA3 via Groq
- 💫 NLP using OpenAi via Sumopod

---

## 🎓 How RAG Works

Retrieval-Augmented Generation (RAG) enhances LLMs by injecting external knowledge. Instead of relying solely on pre-trained data, the model retrieves relevant information from a vector database (like ChromaDB) and uses it to generate accurate, context-aware responses.

---

## 📊 Application Diagram

📄 [Architecture Image](assets/Fast%20API%20RAG.drawio.png)

---

## 🚀 Getting Started Locally

### 1. Clone the Repository

```bash
git clone https://github.com/rizaldisplay/FastApi-RAG.git
cd FASTAPI-RAG
```

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your Groq API Key (.env)
GROQ_API_KEY="your_key_here"
OPENAI_API_KEY="your_key_here"

# Run the FastAPI server
uvicorn main:app --reload
```

## 🌐 API Endpoints (FastAPI)

- `POST /upload/` — Upload PDFs and build vectorstore
- `POST /query/` — Send a query and receive answers
- `DELETE /delete_user_data/` — Delete by user_id on chroma_store
- `DELETE /admin/delete_collection` — Delete all colection data on chroma_store

---


## 🌟 Credits

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/)
- [Sumopod](https://https://sumopod.com/)

---

## ✉️ Contact

For questions or suggestions, open an issue or contact at [afrizal.aminulloh@gmail.com]

---

> Happy Building RAGbots! 🚀