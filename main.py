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
from database import get_connection


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

    try:
        document = fitz.open(
            stream=contents,
            filetype="pdf"
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file"
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

    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    chunks = create_chunks(document_text)

    document_id = str(uuid.uuid4())

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # --------------------------------------------------
        # SAVE DOCUMENT
        # --------------------------------------------------

        cursor.execute(
            """
            INSERT INTO documents (id, filename)
            VALUES (%s, %s)
            """,
            (
                document_id,
                file.filename
            )
        )

        # --------------------------------------------------
        # CREATE EMBEDDINGS + SAVE CHUNKS
        # --------------------------------------------------

        for chunk in chunks:

            embedding = create_embedding(
                client,
                chunk
            )

            embedding_string = "[" + ",".join(
                str(value)
                for value in embedding
            ) + "]"

            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s::vector)
                """,
                (
                    document_id,
                    chunk,
                    embedding_string
                )
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()

    return {
        "id": document_id,
        "filename": file.filename,
        "chunks_count": len(chunks)
    }


# --------------------------------------------------
# GET DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def get_documents():

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                d.id,
                d.filename,
                d.created_at,
                COUNT(c.id) AS chunks_count
            FROM documents d
            LEFT JOIN chunks c
                ON c.document_id = d.id
            GROUP BY
                d.id,
                d.filename,
                d.created_at
            ORDER BY d.created_at DESC
            """
        )

        rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "created_at": row[2],
                "chunks_count": row[3]
            }
            for row in rows
        ]

    finally:
        cursor.close()
        connection.close()


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/documents/{document_id}/ask")
def ask_question(
    document_id: str,
    data: Question
):

    # --------------------------------------------------
    # 1. EMBEDDING QUESTION
    # --------------------------------------------------

    question_embedding = create_embedding(
        client,
        data.question
    )

    # --------------------------------------------------
    # 2. VECTOR SEARCH
    # --------------------------------------------------

    similar_chunks = find_similar_chunks(
        document_id,
        question_embedding,
        top_k=3
    )

    if not similar_chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found or has no chunks"
        )

    # --------------------------------------------------
    # 3. CONTEXT
    # --------------------------------------------------

    context = "\n\n".join(
        chunk["text"]
        for chunk in similar_chunks
    )

    # --------------------------------------------------
    # 4. ASK GEMINI
    # --------------------------------------------------

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

@app.post("/documents/{document_id}/search")
def search_document(
    document_id: str,
    data: Question
):
    # Создаём embedding вопроса
    question_embedding = create_embedding(
        client,
        data.question
    )

    # Ищем похожие chunks в PostgreSQL
    similar_chunks = find_similar_chunks(
        document_id,
        question_embedding,
        top_k=3
    )

    return {
        "question": data.question,
        "results": similar_chunks
    }