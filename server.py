from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_bot(data: Query):
    answer = ask_question(data.question)
    return {"answer": answer}
