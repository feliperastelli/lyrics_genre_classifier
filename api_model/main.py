from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

# Classe para input via JSON
class LyricsInput(BaseModel):
    lyrics: str

# Carrega o modelo treinado
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict_genre(data: LyricsInput):
    prediction = model.predict([data.lyrics])
    return {"genre": prediction[0]}

# Apenas para execução local
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
