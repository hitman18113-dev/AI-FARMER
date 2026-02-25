import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("plant_disease_model.h5", compile=False)

# 🔴 IMPORTANT: Replace with your actual class names
REMEDIES = [
    "Tomato_Late_blight",
    "Tomato_Healthy",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]

# Disease Information Dictionary
DISEASE_INFO = {
    "Tomato_Late_blight": {
        "description": "A serious fungal disease causing dark lesions on leaves and fruit.",
        "prevention": "Avoid overhead watering, ensure proper spacing, use resistant varieties.",
        "remedy": "Apply copper-based fungicides and remove infected plants.",
        "economic": "Can reduce yield by up to 70% if untreated."
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease appearing as yellow spots on upper leaf surface.",
        "prevention": "Improve air circulation and reduce humidity.",
        "remedy": "Use fungicides and remove affected leaves.",
        "economic": "Reduces fruit quality and overall production."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Small circular spots with dark borders on leaves.",
        "prevention": "Avoid overhead irrigation and rotate crops.",
        "remedy": "Apply recommended fungicides.",
        "economic": "Causes early leaf drop reducing yield."
    },
    "Tomato_Healthy": {
        "description": "Plant appears healthy with no visible disease.",
        "prevention": "Maintain proper watering and fertilization.",
        "remedy": "No treatment needed.",
        "economic": "Healthy crops ensure maximum profit."
    }
}

# Streamlit UI
st.set_page_config(page_title="AI Farmer Assistant", layout="wide")

st.title("🌿 AI Farmer Assistant")
st.write("Upload a leaf image to detect plant disease and get detailed insights.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100

    st.subheader("🌱 Prediction Result")
    st.write(f"### 🦠 Disease: {predicted_class}")
    st.write(f"### 📊 Confidence: {confidence:.2f}%")

    # Chart
    st.subheader("📈 Prediction Confidence Chart")
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions * 100)
    plt.xticks(rotation=90)
    plt.ylabel("Confidence (%)")
    st.pyplot(fig)

    # Show detailed info
    if predicted_class in DISEASE_INFO:
        info = DISEASE_INFO[predicted_class]

        st.subheader("📝 Disease Description")
        st.write(info["description"])

        st.subheader("🛡 Prevention Tips")
        st.write(info["prevention"])

        st.subheader("💊 Suggested Remedy")
        st.write(info["remedy"])

        st.subheader("💰 Economic Impact")
        st.write(info["economic"])
    else:
        st.write("No additional information available.")