from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Contexto Clone")

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

class GuessRequest(BaseModel):
    word: str

@app.post("/guess")
def check_guess(payload: GuessRequest):
    guess = payload.word.strip().lower()
    if not guess.isalpha():
        raise HTTPException(status_code=400, detail="Word must contain letters only.")

    guess_embedding = model.encode([guess])
    similarity = float(cosine_similarity(guess_embedding, secret_embedding)[0][0])
    score = round(max(0.0, similarity) * 100, 2)

    return {
        "word": guess,
        "similarity": score,
        "is_correct": guess == SECRET_WORD
    }
