from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model once
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class Feedback(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(feedback: Feedback):
    result = classifier(feedback.text)[0]
    
    label = result["label"]
    score = result["score"]

    priority = "LOW"
    is_negative = False

    if label == "NEGATIVE":
        is_negative = True
        if score > 0.85:
            priority = "HIGH"
        else:
            priority = "MEDIUM"

    return {
        "sentiment": label,
        "confidence": round(score, 4),
        "priority": priority,
        "is_negative": is_negative
    }
