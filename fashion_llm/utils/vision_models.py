from PIL import Image
import numpy as np
import io

# --- Preprocess image for any model ---
def preprocess_image(image_bytes, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Skin tone prediction ---
def predict_skin_tone(image_bytes, skin_model, class_names=['dark', 'lite', 'mid_dark', 'mid_light']):
    img_array = preprocess_image(image_bytes, target_size=(224, 224))
    preds = skin_model.predict(img_array, verbose=0)[0]
    pred_class = class_names[np.argmax(preds)]
    conf = float(np.max(preds))
    all_conf = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    return pred_class, conf, all_conf

# --- Age & Gender prediction (fixed gender flip) ---
def predict_age_gender(image_bytes, age_model, gender_model):
    img_array = preprocess_image(image_bytes, target_size=(128, 128))
    
    # Age prediction
    age_pred = age_model.predict(img_array, verbose=0)
    age_value = float(age_pred[0][0]) if age_pred.ndim > 1 else float(age_pred[0])
    
    # Gender prediction
    gender_pred = gender_model.predict(img_array, verbose=0)
    
    # --- FIXED FLIP LOGIC ---
    if gender_pred.shape[1] == 1:  
        # sigmoid output (e.g., [[0.8]]) means male if high → flip logic
        gender_class = 'female' if gender_pred[0][0] < 0.5 else 'male'
        gender_conf = float(gender_pred[0][0] if gender_class == 'male' else 1 - gender_pred[0][0])
    else:  
        # softmax (e.g., [0.9, 0.1]) → assume index 0=female, 1=male
        gender_class = 'female' if np.argmax(gender_pred[0]) == 0 else 'male'
        gender_conf = float(np.max(gender_pred[0]))
    
    return age_value, gender_class, gender_conf

# --- Optional wrapper for all together ---
def predict_all(image_bytes, skin_model, age_model, gender_model):
    skin_class, skin_conf, skin_all = predict_skin_tone(image_bytes, skin_model)
    age_pred, gender_class, gender_conf = predict_age_gender(image_bytes, age_model, gender_model)
    return {
        "skin_tone": skin_class,
        "skin_conf": skin_conf,
        "skin_all_conf": skin_all,
        "age": age_pred,
        "gender": gender_class,
        "gender_conf": gender_conf
    }
