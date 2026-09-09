import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="BERT Sentiment Analysis API")

# Explicit BERT base model fine-tuned on SST-2
MODEL_NAME = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class TextRequest(BaseModel):
    text: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
def analyze_sentiment(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Tokenize input sequence using BERT Tokenizer
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Perform forward pass through BERT model using PyTorch
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probabilities, dim=-1)

    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    predicted_label = label_map[prediction.item()]
    confidence_score = round(float(confidence.item()), 4)

    return {"label": predicted_label, "score": confidence_score}
