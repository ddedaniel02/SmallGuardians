from fastapi import FastAPI, Request, HTTPException
from sklearn import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
import torch
import os

app = FastAPI()

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8001")
PROMPT_FIELD = os.getenv("PROMPT_FIELD", "prompt")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-Prompt-Guard-2-86M")
CLASSIFICATION_LABEL= os.getenv("CLASSIFICATION_LABEL", "LABEL_1")
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", 0.7))
TOKEN=os.getenv("TOKEN", "")

def is_malicious(prompt: str) -> bool:
    pipe = pipeline("text-classification", model=MODEL_NAME, token=TOKEN)
    classification = pipe(prompt)
    print(classification)
    if classification[0]["label"] == CLASSIFICATION_LABEL and classification[0]["score"] >= CLASSIFICATION_THRESHOLD:
        return True
    return False

@app.api_route("/{path:path}", methods=["POST", "GET", "PUT", "DELETE"])
async def waf_proxy(request: Request, path: str):
    method = request.method
    headers = dict(request.headers)
    url = f"{BACKEND_BASE_URL}/{path}"

    data = {}
    if method in ["POST", "PUT"]:
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        prompt = data.get(PROMPT_FIELD, "")
        if prompt and is_malicious(prompt):
            raise HTTPException(status_code=403, detail="Forbidden: Malicious content detected")

    try:
        response = requests.request(method, url, headers=headers, json=data if data else None, timeout=10)
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")
