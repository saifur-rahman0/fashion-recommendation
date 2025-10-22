import requests
import os
from dotenv import load_dotenv

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_images(query, num_results=3):
    """
    Search outfit images using Serper.dev API.
    """
    url = "https://google.serper.dev/images"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Serper API error:", response.text)
        return []

    data = response.json()
    return [img.get("imageUrl") for img in data.get("images", []) if img.get("imageUrl")]
