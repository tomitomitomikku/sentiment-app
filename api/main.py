from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = joblib.load(os.path.join(BASE_DIR, 'models/model.joblib'))
    ml_models["vectorizer"] = joblib.load(os.path.join(BASE_DIR, 'models/vectorizer.joblib'))
    print("モデルのロード完了！")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(input: TextInput):
    tfidf = ml_models["vectorizer"].transform([input.text])
    prediction = ml_models["model"].predict(tfidf)[0]
    sentiment = 1 if prediction == "positive" else 0
    return {"text": input.text, "sentiment": sentiment}
