# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import os
import requests
from dotenv import load_dotenv

# === Local imports ===
from utils.vision_models import predict_all
from fashion_agent import fashion_recommendation

# === Load models ===
SKIN_MODEL_PATH = "../Models/skin_tone_final_tf.keras"
AGE_MODEL_PATH = "../Models/age_vgg.h5"
GENDER_MODEL_PATH = "../Models/gender_resnet.h5"

skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH)
age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)

# === Environment setup ===
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# === FastAPI App ===
app = FastAPI(title="Fashion Vision API")


# ========== 🔍 SERPER SEARCH TEST ROUTE ==========
@app.get("/search")
def search_images_endpoint(query: str, num_results: int = 3):
    """
    Simple endpoint to test Serper image search.
    Example: /search?query=summer+outfit
    """
    try:
        if not SERPER_API_KEY:
            raise ValueError("Missing SERPER_API_KEY in .env")

        url = "https://google.serper.dev/images"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if response.status_code != 200:
            return {"error": f"Serper API error: {data}"}

        images = [item["imageUrl"] for item in data.get("images", [])[:num_results]]
        return {"query": query, "images": images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 🧠 RECOMMENDATION PIPELINE ==========
@app.post("/recommend")
async def recommend_outfit(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...)
):

    try:
        image_bytes = await file.read()
        result = fashion_recommendation(image_bytes, lat, lon, prompt, skin_model, age_model, gender_model)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== BASIC PREDICT ROUTE ==========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_all(image_bytes, skin_model, age_model, gender_model)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== HOME ==========
@app.get("/")
def home():
    return {"message": "Fashion Vision API running.", "routes": ["/predict", "/recommend", "/search"]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
