from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_question   # rag_pipeline.py must be in same folder

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Women Law RAG API is running. Use POST /ask to query."}

@app.post("/ask")
def ask_bot(data: Query):
    return {"answer": ask_question(data.question)}
