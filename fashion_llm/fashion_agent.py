import json
from utils.weather_utils import get_weather
from utils.serper_search import search_images
from utils.vision_models import predict_all
from fashion_llm import generate_detailed_outfit  # ✅ Import the Gemini function


def fashion_recommendation(image_bytes, lat, lon, prompt, skin_model, age_model, gender_model):
    # --- Step 1: Vision analysis ---
    vision = predict_all(image_bytes, skin_model, age_model, gender_model)
    skin_tone = vision["skin_tone"]
    age = int(vision["age"])
    gender = vision["gender"]

    # --- Step 2: Weather detection ---
    weather = get_weather(lat, lon)
    temp = weather.get("temp", "N/A")
    desc = weather.get("desc", "N/A")
    season = weather.get("season", "N/A")

    # --- Step 3: Create context prompt for Gemini ---
    full_prompt = (
        f"A {age}-year-old {gender} with {skin_tone} skin tone. "
        f"The weather is {temp}°C, {desc}, during {season}. "
        f"Occasion: {prompt}. Suggest a luxurious, stylish outfit in JSON format."
    )

    # --- Step 4: Get Gemini outfit suggestion ---
    outfit_json = generate_detailed_outfit(full_prompt, gender)

    # --- Step 5: Search outfit images ---
    outfit_images = {}
    if "error" not in outfit_json:
        for key, val in outfit_json.items():
            if isinstance(val, str) and len(val.strip()) > 3:
                query = f"{val} outfit for {gender}"
                try:
                    images = search_images(query, num_results=3)
                    outfit_images[key] = images
                except Exception as e:
                    outfit_images[key] = [f"Image fetch error: {str(e)}"]
    else:
        outfit_images = {"error": "Gemini failed to produce valid JSON"}

    # --- Step 6: Return final structured response ---
    return {
        "vision_results": vision,
        "weather": {"temp": temp, "desc": desc, "season": season},
        "outfit_suggestions": outfit_json,
        "outfit_images": outfit_images,
    }
