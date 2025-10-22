# utils/weather_utils.py
import requests
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(lat: float, lon: float):
    """
    Get temperature, weather condition, and season from OpenWeather API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
    
    Returns:
        dict: {"temp": float, "condition": str, "season": str} or {"error": str}
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()

        if res.status_code != 200:
            return {"error": data.get("message", "Weather fetch failed")}

        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"].title()

        # Determine season by temperature (simplified)
        if temp > 30:
            season = "summer"
        elif temp > 20:
            season = "spring"
        elif temp > 10:
            season = "autumn"
        else:
            season = "winter"

        return {"temp": temp, "condition": condition, "season": season}

    except Exception as e:
        return {"error": f"Weather API exception: {str(e)}"}
