# Sentiment Analysis Using Pretrained BERT Model

## Aim
To implement a Sentiment Analysis web application using a pretrained BERT model (DistilBERT fine-tuned on SST-2) with FastAPI backend and HTML/CSS/JS frontend.

## Theory
Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) developed by Google. Standard BERT is trained bidirectionally to understand context from both left-to-right and right-to-left.

For Sentiment Analysis, a pretrained transformer model processes input tokens, passes them through self-attention layers, and produces a feature representation used by a classification head to output the probability distribution over sentiment labels (e.g., POSITIVE or NEGATIVE).

## Software and Hardware Requirements
### Software Requirements
- **Operating System**: Linux / macOS / Windows
- **Programming Language**: Python 3.10+
- **Key Libraries**: `fastapi`, `uvicorn`, `transformers`, `torch`, `jinja2`
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API)

### Hardware Requirements
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher (Apple Silicon / x86_64)
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: ~2 GB free disk space for PyTorch and pretrained model weights
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster inference (CPU execution supported)

## Mathematical Formula
1. **Self-Attention Mechanism**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   where $Q$, $K$, and $V$ represent Query, Key, and Value matrices, and $d_k$ is the dimension of keys.

2. **Softmax Output Classification**:
   $$P(y = c | x) = \frac{e^{z_c}}{\sum_{j} e^{z_j}}$$
   where $z_c$ is the logit output for sentiment class $c \in \{\text{POSITIVE}, \text{NEGATIVE}\}$.

## Libraries Used
- **`transformers`**: Hugging Face library providing access to pretrained DistilBERT models.
- **`torch`**: PyTorch tensor computation and deep learning framework.
- **`fastapi`**: Modern, fast web framework for building APIs with Python.
- **`uvicorn`**: ASGI server implementation for hosting FastAPI.

## Project Structure
```text
.
├── app.py
├── static/
│   ├── index.html
│   └── style.css
├── test_app.py
└── README.md
```

## Sample Code Overview (`app.py`)
```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class TextRequest(BaseModel):
    text: str

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/analyze")
def analyze_sentiment(request: TextRequest):
    result = classifier(request.text)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}
```

## Sample Output
- **Input Text**: `"I absolutely love working with modern transformer models!"`
- **Result**: `POSITIVE` (Confidence Score: `0.9999`)
- **Input Text**: `"This experience was disappointing and frustrating."`
- **Result**: `NEGATIVE` (Confidence Score: `0.9998`)

## Conclusion
The Sentiment Analysis web application was successfully implemented using a pretrained DistilBERT model integrated with FastAPI and a modern HTML/CSS interface.

- **Name**: zain pawle
- **Roll No**: 25dco08
- **Batch**: 3
- **Class**: TECO1
