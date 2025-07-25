# main.py

from fastapi import FastAPI, UploadFile, File, Form, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List
from modules.load_vectorstore import load_vectorstore
from modules.rag_setup import load_components, create_rag_chain
from logger import logger
import config # Impor config

app = FastAPI(title="RAGBOT-Advanced")

# 🧠 RAG_COMPONENTS akan diisi saat startup
RAG_COMPONENTS: dict = {}

@app.on_event("startup")
def startup_event():
    vectorstore, llm, embedding_function = load_components()
    RAG_COMPONENTS['vectorstore'] = vectorstore
    RAG_COMPONENTS['llm'] = llm
    RAG_COMPONENTS['embedding_function'] = embedding_function # Simpan embedding_function

# Middleware (sudah bagus)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error":str(exc)})

# --- Endpoint Utama ---
@app.post("/upload/")
async def upload_documents(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} files from user {user_id}")
        # Gunakan embedding function dari memori
        embedding_function = RAG_COMPONENTS['embedding_function']
        load_vectorstore(files, user_id=user_id, embedding_function=embedding_function)
        
        logger.info(f"Documents added to Chroma for user {user_id}")
        return {"message": "Files processed and vectorstore updated successfully."}
    except Exception as e:
        logger.exception("Error during document upload")
        return JSONResponse(status_code=500, content={"error": str(e)})

class QueryRequest(BaseModel):
    question: str
    user_id: str

@app.post("/query/")
async def handle_query(request_data: QueryRequest):
    try:
        question = request_data.question
        user_id = request_data.user_id
        
        logger.info(f"[user {user_id}] query: {question}")
        
        # Ambil komponen dari memori
        vectorstore = RAG_COMPONENTS['vectorstore']
        llm = RAG_COMPONENTS['llm']

        # Buat retriever on-the-fly dengan filter per user (sangat efisien)
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': config.RETRIEVER_K, 'filter': {'user_id': user_id}}
        )

        # ---- PERUBAHAN DI SINI ----

        # 1. Dapatkan dokumen sumber terlebih dahulu
        source_documents = retriever.invoke(question)

        # 2. Buat RAG chain
        rag_chain = create_rag_chain(llm, retriever)

        # 3. Panggil RAG chain dengan pertanyaan
        answer = await rag_chain.ainvoke(question) # Menggunakan 'ainvoke' untuk async

        logger.info("Query successful")
        
        # 4. Kembalikan jawaban dan sumber yang sudah kita dapatkan
        return {
            "answer": answer,
            "source_documents": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                }
                for doc in source_documents
            ]
        }
    except Exception as e:
        logger.exception("Error processing query")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
# Buat Pydantic model untuk menerima user_id
class UserDataRequest(BaseModel):
    user_id: str

@app.delete("/delete_user_data/")
async def delete_user_data(request_data: UserDataRequest):
    """
    Menghapus semua dokumen yang terkait dengan user_id tertentu dari vectorstore.
    """
    try:
        user_id = request_data.user_id
        logger.info(f"Attempting to delete all data for user_id: {user_id}")

        vectorstore = RAG_COMPONENTS['vectorstore']

        # Langsung akses koleksi internal ChromaDB untuk menggunakan filter 'where'.
        # Ini adalah cara yang paling andal untuk menghapus berdasarkan metadata.
        vectorstore._collection.delete(where={"user_id": user_id})

        # Penting: Lakukan persist untuk menyimpan perubahan setelah penghapusan.
        vectorstore.persist()

        logger.info(f"Successfully deleted data for user_id: {user_id}")
        return {"message": f"All documents for user_id '{user_id}' have been deleted."}

    except Exception as e:
        logger.exception("Error during user data deletion")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/admin/delete_collection/")
async def delete_collection():
    try:
        vectorstore = RAG_COMPONENTS['vectorstore']
        vectorstore.delete_collection()
        logger.info("Vectorstore collection deleted successfully. Restart the app to re-create.")
        # Re-initialize after deletion if needed, or simply notify
        RAG_COMPONENTS['vectorstore']._collection = None # Invalidate in-memory collection
        return {"message": "Vectorstore collection deleted. Please restart the application."}
    except Exception as e:
        logger.exception("Error deleting vectorstore collection")
        return JSONResponse(status_code=500, content={"error": str(e)})