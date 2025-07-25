# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# DIRECTORY CONFIGS
PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploaded_pdfs"

# EMBEDDING MODEL CONFIG
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
EMBEDDING_MODEL_KWARGS = {"device": "cpu"} # Sesuaikan jika pakai GPU
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": True}

# LLM CONFIG (Pilih salah satu)
# Gunakan variabel environment untuk memilih LLM, lebih fleksibel
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq") # 'groq' atau 'openai'

# GROQ CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama3-8b-8192" # ganti model di sini

# OPENAI/SUMOPOD CONFIG
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://ai.sumopod.com/v1"
OPENAI_MODEL_NAME = "gpt-4o-mini"

# RAG CONFIG
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 100
RETRIEVER_K = 3