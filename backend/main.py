from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI(title="WordNear")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready!")

SECRET_WORD = "ocean"
secret_embedding = model.encode([SECRET_WORD])

# Serve frontend index.html on root route "/"
FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

@app.get("/")
def serve_home():
    return FileResponse(FRONTEND_PATH)

class GuessRequest(BaseModel):
    word: str

@app.post("/guess")
def check_guess(payload: GuessRequest):
    guess = payload.word.strip().lower()
    if not guess.isalpha():
        raise HTTPException(status_code=400, detail="Word must contain letters only.")

    is_correct = (guess == SECRET_WORD)
    if is_correct:
        score = 100.0
    else:
        guess_embedding = model.encode([guess])
        similarity = float(cosine_similarity(guess_embedding, secret_embedding)[0][0])
        score = round(max(0.0, similarity) * 100, 2)

    return {
        "word": guess,
        "similarity": score,
        "is_correct": is_correct
    }