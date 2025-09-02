from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 軽い日本語要約モデル
summarizer = pipeline("summarization", model="sonoisa/t5-small-japanese")

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: TextRequest):
    result = summarizer(
        request.text,
        max_length=100,
        min_length=20,
        do_sample=False
    )
    return {"summary": result[0]["summary_text"]}
