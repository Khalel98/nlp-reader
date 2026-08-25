import os

import fitz
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from chunking import create_chunks
from embeddings import create_embedding
from search import find_similar_chunks

from repository import (
    create_document,
    create_chunk,
    get_document_chunks,
)


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


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    # 1. Читаем PDF
    contents = await file.read()

    document = fitz.open(
        stream=contents,
        filetype="pdf"
    )

    # 2. Извлекаем текст
    document_text = ""

    for page in document:
        document_text += page.get_text()

    document.close()

    # 3. Разбиваем текст на chunks
    chunks = create_chunks(
        document_text
    )

    # 4. Создаём запись документа в PostgreSQL
    document_id = create_document(
        file.filename
    )

    # 5. Создаём embedding каждого chunk
    #    и сохраняем chunk + embedding в PostgreSQL
    for chunk in chunks:

        embedding = create_embedding(
            client,
            chunk
        )

        create_chunk(
            document_id,
            chunk,
            embedding
        )

    return {
        "document_id": str(document_id),
        "filename": file.filename,
        "chunks_count": len(chunks)
    }


@app.post("/documents/{document_id}/ask")
def ask_question(
    document_id: str,
    data: Question
):
    # 1. Получаем chunks конкретного документа
    document_chunks = get_document_chunks(
        document_id
    )

    if not document_chunks:
        return {
            "error": "Document not found or has no chunks"
        }

    # 2. Создаём embedding вопроса
    question_embedding = create_embedding(
        client,
        data.question
    )

    # 3. Ищем наиболее похожие chunks
    similar_chunks = find_similar_chunks(
        question_embedding,
        document_chunks,
        top_k=3
    )

    # 4. Если ничего достаточно похожего нет
    if not similar_chunks:
        return {
            "document_id": document_id,
            "question": data.question,
            "answer": "В документе нет информации по этому вопросу."
        }

    # 5. Собираем найденные chunks
    context = "\n\n".join(
        chunk["text"]
        for chunk in similar_chunks
    )

    # 6. Формируем prompt
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

    # 7. Отправляем контекст + вопрос Gemini
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return {
        "document_id": document_id,
        "question": data.question,
        "answer": response.text
    }