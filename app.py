import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from disease_info import disease_info


# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []


# Load Model
model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)


# Class Names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


# Page Title
st.title("🌱 Plant Disease Detection AI")

st.markdown(
    """
    ### AI-Based Plant Disease Classification System

    This application uses **Convolutional Neural Network (CNN)**
    to detect plant diseases from leaf images.

    Upload a leaf image and get:
    - 🌿 Disease Prediction
    - 📊 Confidence Score
    - 💊 Treatment Suggestions
    - 🛡️ Prevention Tips
    """
)


# Sidebar Project Info
st.sidebar.title("🌱 Project Info")

st.sidebar.write(
    """
    **Project:** Plant Disease Detection

    **Technology:**
    - Python
    - TensorFlow
    - CNN
    - Streamlit

    **Dataset:**
    PlantVillage

    **Classes:**
    15 Plant Disease Categories
    """
)
# Prediction History Sidebar
st.sidebar.subheader("📌 Prediction History")
if len(st.session_state.history) > 0:
    for item in st.session_state.history:
        st.sidebar.write("🌿", item["Disease"])
        st.sidebar.write(
            "Confidence:",
            item["Confidence"]
        )
        st.sidebar.write("---")
else:
    st.sidebar.write("No predictions yet")
# Upload Image
uploaded_file = st.file_uploader(
    "📷 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(
        img,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )
    if st.button("Predict Disease"):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(
            img_array,
            axis=0
        )
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        disease = class_names[index]
        # Save Prediction History
        st.session_state.history.append(
            {
                "Disease": disease_info[disease]["name"]
                if disease in disease_info
                else disease.replace("_", " "),

                "Confidence": f"{confidence:.2f}%"
            }
        )
        st.info(f"Confidence: {confidence:.2f}%")
        if disease in disease_info:
            st.success(f"🌿 Disease: {disease_info[disease]['name']}")
            st.subheader("Symptoms")
            st.write(disease_info[disease]["symptoms"])
            st.subheader("Treatment")
            for item in disease_info[disease]["treatment"]:
                st.write("✅", item)
            st.subheader("Prevention")
            for item in disease_info[disease]["prevention"]:
                st.write("🛡️", item)
        else:
            st.success(f"🌿 Disease: {disease.replace('_',' ')}")
            st.warning("Detailed treatment information will be added soon.")