from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(os.path.join(BASE_DIR, '../models/model.joblib'))
    vectorizer = joblib.load(os.path.join(BASE_DIR, '../models/vectorizer.joblib'))
except Exception as e:
    model = None
    vectorizer = None

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(input: TextInput):
    if model is None or vectorizer is None:
        return {"error": "モデルが読み込めませんでした"}
    tfidf = vectorizer.transform([input.text])
    prediction = model.predict(tfidf)[0]
    sentiment = 1 if prediction == "positive" else 0
    return {"text": input.text, "sentiment": sentiment}
