import os

import fitz
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from chunking import create_chunks
from embeddings import create_embedding
from search import find_similar_chunks

load_dotenv()


app = FastAPI()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


document_chunks = []


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Document AI is running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_chunks

    contents = await file.read()

    document = fitz.open(
        stream=contents,
        filetype="pdf"
    )

    document_text = ""

    for page in document:
        document_text += page.get_text()

    document.close()

    chunks = create_chunks(document_text)

    document_chunks = []

    for chunk in chunks:
        embedding = create_embedding(
            client,
            chunk
        )

        document_chunks.append({
            "text": chunk,
            "embedding": embedding
        })

    return {
        "filename": file.filename,
        "chunks_count": len(document_chunks)
    }


@app.post("/ask")
def ask_question(data: Question):
    if not document_chunks:
        return {
            "error": "Please upload a document first"
        }

    # 1. Создаём embedding вопроса
    question_embedding = create_embedding(
        client,
        data.question
    )

    # 2. Ищем самые подходящие куски документа
    similar_chunks = find_similar_chunks(
        question_embedding,
        document_chunks,
        top_k=3
    )

    # 3. Собираем найденные куски в один текст
    context = "\n\n".join(
        chunk["text"]
        for chunk in similar_chunks
    )

    # 4. Отправляем вопрос + найденный контекст Gemini
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
        "question": data.question,
        "answer": response.text
    }