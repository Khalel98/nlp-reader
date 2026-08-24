import os

import fitz
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel


load_dotenv()


app = FastAPI()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


document_text = ""


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Document AI is running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global document_text

    contents = await file.read()

    document = fitz.open(
        stream=contents,
        filetype="pdf"
    )

    document_text = ""

    for page in document:
        document_text += page.get_text()

    document.close()

    return {
        "filename": file.filename,
        "message": "Document uploaded successfully"
    }


@app.post("/ask")
def ask_question(data: Question):
    if not document_text:
        return {
            "error": "Please upload a document first"
        }

    prompt = f"""
Ты отвечаешь на вопросы по документу.

Используй только информацию из документа.

Если ответа в документе нет, скажи:
"В документе нет информации по этому вопросу."

Документ:
{document_text}

Вопрос:
{data.question}
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return {
        "answer": response.text
    }