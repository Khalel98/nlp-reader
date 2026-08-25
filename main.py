import os
import uuid

import fitz
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from chunking import create_chunks
from embeddings import create_embedding
from search import find_similar_chunks
from storage import load_documents, save_documents, get_document


load_dotenv()


app = FastAPI()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Document AI is running"
    }


# --------------------------------------------------
# UPLOAD DOCUMENT
# --------------------------------------------------

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    contents = await file.read()

    document = fitz.open(
        stream=contents,
        filetype="pdf"
    )

    document_text = ""

    for page in document:
        document_text += page.get_text()

    document.close()

    if not document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF does not contain text"
        )

    # Разбиваем документ на chunks
    chunks = create_chunks(document_text)

    # Уникальный ID документа
    document_id = str(uuid.uuid4())

    document_chunks = []

    # Создаём embedding для каждого chunk
    for chunk in chunks:

        embedding = create_embedding(
            client,
            chunk
        )

        document_chunks.append({
            "text": chunk,
            "embedding": embedding
        })

    # Формируем документ
    new_document = {
        "id": document_id,
        "filename": file.filename,
        "chunks": document_chunks
    }

    # Загружаем существующие документы
    documents = load_documents()

    # Добавляем новый
    documents.append(new_document)

    # Сохраняем
    save_documents(documents)

    return {
        "id": document_id,
        "filename": file.filename,
        "chunks_count": len(document_chunks)
    }


# --------------------------------------------------
# GET DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def get_documents():

    documents = load_documents()

    return [
        {
            "id": document["id"],
            "filename": document["filename"],
            "chunks_count": len(document["chunks"])
        }
        for document in documents
    ]


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/documents/{document_id}/ask")
def ask_question(
    document_id: str,
    data: Question
):
    # 1. Превращаем вопрос в embedding
    question_embedding = create_embedding(
        client,
        data.question
    )

    # 2. Ищем похожие chunks непосредственно в PostgreSQL
    similar_chunks = find_similar_chunks(
        document_id,
        question_embedding,
        top_k=3
    )

    if not similar_chunks:
        return {
            "error": "Document not found or has no chunks"
        }

    # 3. Собираем контекст
    context = "\n\n".join(
        chunk["text"]
        for chunk in similar_chunks
    )

    # 4. Отправляем контекст + вопрос Gemini
    prompt = f"""
Ты отвечаешь на вопросы по документу.

Используй только информацию из предоставленного контекста.

Если ответа в контексте нет, скажи:
"В документе нет информации по этому вопросу."

Контекст документа:

{context}

Вопрос:

{data.question}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return {
        "document_id": document_id,
        "question": data.question,
        "answer": response.text
    }