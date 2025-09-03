from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# 軽量モデルに変更
MODEL_NAME = "rinna/japanese-gpt-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/tmp")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="/tmp")

summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize(request: TextRequest):
    result = summarizer(
        request.text,
        max_length=100,   # 出力の長さを制限
        do_sample=False   # サンプリングせず決定的に生成
    )
    return {"summary": result[0]["generated_text"]}

