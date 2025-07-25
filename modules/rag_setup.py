# modules/rag_setup.py

from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import config # Impor config

def load_components():
    """Memuat semua komponen berat sekali saja saat startup."""
    print("🚀 Loading RAG components...")
    
    # 1. Muat Embedding Model sekali
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs=config.EMBEDDING_MODEL_KWARGS,
        encode_kwargs=config.EMBEDDING_ENCODE_KWARGS
    )
    
    # 2. Inisialisasi koneksi ke Vectorstore dari disk sekali
    print(f"Connecting to vectorstore at: {config.PERSIST_DIR}")
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embedding_function
    )
    
    # 3. Muat LLM sekali berdasarkan konfigurasi
    llm = None
    if config.LLM_PROVIDER == "groq":
        print(f"Loading Groq LLM: {config.GROQ_MODEL_NAME}")
        llm = ChatGroq(
            temperature=0.7,
            model_name=config.GROQ_MODEL_NAME,
            api_key=config.GROQ_API_KEY 
        )
    elif config.LLM_PROVIDER == "openai":
        print(f"Loading OpenAI LLM: {config.OPENAI_MODEL_NAME}")
        llm = ChatOpenAI(
            model_name=config.OPENAI_MODEL_NAME,
            temperature=0.7,
            max_tokens=500,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL
        )
    else:
        raise ValueError(f"LLM provider '{config.LLM_PROVIDER}' not supported.")

    print("✅ RAG components loaded successfully.")
    return vectorstore, llm, embedding_function # Kembalikan juga embedding_function

# Template prompt yang menggabungkan pencarian dan pemolesan jawaban
PROMPT_TEMPLATE = """
Kamu adalah customer service profesional, asisten virtual yang profesional, ramah, dan solutif.
Gunakan informasi dari konteks yang diberikan di bawah untuk menjawab pertanyaan pengguna.

📌 **Aturan Penting:**
1.  Saat membandingkan harga atau angka, selalu lakukan perbandingan matematis. Jika budget pengguna lebih besar atau sama dengan harga paket, maka tawarkan paket tersebut sebagai solusi yang memungkinkan.
2.  **Jawaban Akurat:** Hanya gunakan informasi dari "Konteks". Jika jawaban tidak ada di sana, katakan dengan sopan: "Maaf, informasi mengenai hal tersebut belum tersedia saat ini." JANGAN mengarang jawaban.
3.  **Bahasa Hangat:** Gunakan bahasa Indonesia yang natural, hangat, dan mudah dipahami. Sapa pengguna dengan ramah.
4.  **Format Rapi:** Jika ada daftar atau langkah-langkah, gunakan poin atau nomor.
5.  **Proaktif:** Akhiri jawaban dengan kalimat positif yang mengundang diskusi lebih lanjut, seperti "Semoga membantu ya! Ada lagi yang bisa saya bantu? 😊"
6.  **Pesan Ringkas:** tolong rangkum lagi menjadi kalimat yang mudah di baca di pesan WhatsApp.

---
**Konteks:**
{context}
---

**Pertanyaan Pengguna:**
{question}

**Jawabanmu:**
"""

def create_rag_chain(llm, retriever):
    """
    Membuat RAG chain menggunakan LCEL (LangChain Expression Language).
    Ini adalah cara modern yang lebih stabil.
    """
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Definisikan alur RAG menggunakan LCEL
    rag_chain = (
        # RunnableParallel mengambil input pertanyaan dan secara paralel:
        # 1. Mengambil dokumen (context) menggunakan retriever.
        # 2. Meneruskan pertanyaan (question) tanpa perubahan.
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt          # Memasukkan hasil ke dalam prompt
        | llm             # Memasukkan prompt yang sudah diformat ke LLM
        | StrOutputParser() # Mengambil output string dari LLM
    )

    return rag_chain