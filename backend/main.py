from datetime import date
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Wordnear API")

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

# Curated secret word bank
WORD_BANK = [
    "ocean", "planet", "guitar", "forest", "shadow", 
    "castle", "whisper", "desert", "bridge", "flame",
    "meadow", "harbor", "valley", "canyon", "breeze"
]

# Daily rotation: changes deterministically each day at midnight
today_index = date.today().toordinal() % len(WORD_BANK)
SECRET_WORD = WORD_BANK[today_index]
secret_embedding = model.encode([SECRET_WORD])
print(f"Loaded target secret word for today: '{SECRET_WORD}'")


class GuessRequest(BaseModel):
    word: str

@app.post("/new-game")
def start_new_game():
    global SECRET_WORD, secret_embedding
    available_words = [w for w in WORD_BANK if w != SECRET_WORD]
    SECRET_WORD = random.choice(available_words)
    secret_embedding = model.encode([SECRET_WORD])
    print(f"New secret word selected: '{SECRET_WORD}'")
    return {"status": "ok", "message": "New game started"}
@app.post("/guess")
def check_guess(payload: GuessRequest):
    guess = payload.word.strip().lower()
    if not guess.isalpha():
        raise HTTPException(status_code=400, detail="Word must contain letters only.")

    guess_embedding = model.encode([guess])
    similarity = float(cosine_similarity(guess_embedding, secret_embedding)[0][0])
    score = round(max(0.0, similarity) * 100, 2)
    is_correct = (guess == SECRET_WORD)

    return {
        "word": guess,
        "similarity": score,
        "is_correct": is_correct
    }