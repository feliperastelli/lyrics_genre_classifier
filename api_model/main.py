from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
from pathlib import Path

# Classe para input via JSON
class LyricsInput(BaseModel):
    lyrics: str

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "model.pkl"

# Carrega o modelo treinado
with open(model_path , "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict_genre(data: LyricsInput):
    prediction = model.predict([data.lyrics])
    return {"genre": prediction[0]}

# Apenas para execução local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
