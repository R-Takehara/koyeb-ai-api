from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

MODEL_NAME = "rinna/japanese-gpt-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/tmp")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="/tmp")

summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: TextRequest):
    result = summarizer(request.text, max_length=100, do_sample=False)
    return {"summary": result[0]["generated_text"]}


