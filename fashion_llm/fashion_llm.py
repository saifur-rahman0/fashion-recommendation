from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

def generate_detailed_outfit(prompt: str, gender: str):
    """
    Generate a structured outfit suggestion (top, bottom, shoes, watch, necklace if female).
    """
    full_prompt = f"""
    You are a luxury AI fashion stylist. Based on this context, suggest a complete outfit for a {gender}.
    Context: {prompt}

    Output a JSON object with:
    - top
    - bottom
    - shoes
    - watch
    Also include "necklace" if the gender is female.

    Example JSON:
    {{
      "top-outfit": "Silk white blouse",
      "bottom-outfit": "High-waisted beige trousers",
      "shoes": "Nude heels",
      "watch": "Minimal gold watch",
      "necklace": "Pearl pendant"
    }}
    """

    message = HumanMessage(content=full_prompt)
    response = model.invoke([message])
    content = response.content.strip()

    # 🧹 Clean and parse
    cleaned = clean_json_output(content)
    try:
        outfit_json = json.loads(cleaned)
    except json.JSONDecodeError:
        outfit_json = {"error": "Failed to parse Gemini response", "raw": content}

    print(outfit_json)
    return outfit_json



def clean_json_output(text: str):
    """
    Clean Gemini's markdown-wrapped JSON output.
    Example input: ```json { "top": "..."} ```
    """
    # Remove code fences if present
    cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    return cleaned


