import streamlit as st
import tensorflow as tf
from PIL import Image
import io

from fashion_agent import fashion_recommendation
from utils.vision_models import predict_all

# -----------------------------
# 🧠 Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    skin_model = tf.keras.models.load_model("../Models/skin_tone_final_tf.keras")
    age_model = tf.keras.models.load_model("../Models/age_vgg.h5", compile=False)
    gender_model = tf.keras.models.load_model("../Models/gender_resnet.h5", compile=False)
    return skin_model, age_model, gender_model


skin_model, age_model, gender_model = load_models()

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------
st.set_page_config(page_title="Luxury AI Stylist", layout="wide")
st.title("👗 Luxury AI Stylist – Streamlit Edition")

st.markdown(
    "Upload a photo and get AI-powered fashion recommendations based on your **skin tone**, **age**, "
    "**gender**, and **current weather conditions**. 🌤️"
)

# --- Input fields ---
prompt = st.text_input("✨ Describe your occasion or style preference:", placeholder="e.g., Beach party, business meeting, evening date...")
lat = st.number_input("📍 Latitude", value=23.8103, format="%.4f")
lon = st.number_input("📍 Longitude", value=90.4125, format="%.4f")
uploaded_file = st.file_uploader("📸 Upload your photo", type=["jpg", "jpeg", "png"])

execute = st.button("🚀 Generate Outfit Suggestions")

# -----------------------------
# 🔍 When Button Clicked
# -----------------------------
if execute:
    if not uploaded_file:
        st.warning("⚠️ Please upload an image first.")
    elif not prompt:
        st.warning("⚠️ Please enter a style prompt.")
    else:
        with st.spinner("Analyzing your image and generating recommendations..."):
            # --- Read image bytes ---
            image_bytes = uploaded_file.read()

            # --- Vision prediction ---
            vision_result = predict_all(image_bytes, skin_model, age_model, gender_model)
            st.subheader("🧠 Vision Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Skin Tone", vision_result["skin_tone"])
            col2.metric("Predicted Age", f"{int(vision_result['age'])}")
            col3.metric("Gender", vision_result["gender"].capitalize())

            # --- Show uploaded image ---
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Uploaded Photo", use_container_width=True)

            # --- Generate Outfit Recommendations ---
            result = fashion_recommendation(
                image_bytes, lat, lon, prompt, skin_model, age_model, gender_model
            )

            # --- Weather info ---
            st.subheader("🌦️ Current Weather")
            weather = result.get("weather", {})
            st.write(
                f"**Temperature:** {weather.get('temp', 'N/A')}°C | "
                f"**Season:** {weather.get('season', 'N/A').capitalize()} | "
                f"**Description:** {weather.get('desc', 'N/A').capitalize()}"
            )

            # --- Outfit suggestions ---
            outfit_data = result.get("outfit_suggestions", {})
            st.subheader("👕 Outfit Recommendations")

            for key, value in outfit_data.items():
                st.markdown(f"**{key.capitalize()}**: {value}")

                # --- Images for each item ---
                image_urls = result.get("outfit_images", {}).get(key, [])
                if image_urls:
                    cols = st.columns(len(image_urls))
                    for i, img_url in enumerate(image_urls):
                        cols[i].image(img_url, use_container_width=True)
                else:
                    st.write("_No images found for this item._")

st.markdown("---")
st.caption("© 2025 Luxury AI Stylist — Powered by Gemini + Serper + TensorFlow")
