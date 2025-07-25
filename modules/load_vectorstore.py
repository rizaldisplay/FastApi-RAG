# modules/load_vectorstore.py

import os
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config # Impor config

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

def load_vectorstore(uploaded_files, user_id: str, embedding_function): # Terima embedding_function
    """Menyimpan file, memproses, dan menambahkannya ke vectorstore."""
    file_paths = []
    
    for file in uploaded_files:
        save_path = Path(config.UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
        
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE, 
        chunk_overlap=config.RAG_CHUNK_OVERLAP
    )
    texts = splitter.split_documents(docs)
    
    # Tambahkan user_id ke metadata setiap dokumen
    for doc in texts:
        doc.metadata["user_id"] = user_id
        # Tambahan: Simpan juga nama file sumber
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    # Gunakan embedding_function yang sudah ada, jangan buat baru
    vectorstore = Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embedding_function
    )
    vectorstore.add_documents(texts)
    # Persist sudah otomatis di-handle oleh Chroma saat inisialisasi dan penambahan, 
    # namun memanggilnya secara eksplisit tidak ada salahnya.
    vectorstore.persist() 
    
    return vectorstore