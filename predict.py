import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
# Load Model
model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)
# Classes
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
# Image Path
img_path = "test.JPG"
# Load Image
img = image.load_img(
    img_path,
    target_size=(224,224)
)
# Convert Image
img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)
index = np.argmax(prediction)
confidence = np.max(prediction) * 100
print("Disease:", class_names[index])
print("Confidence:", round(confidence,2), "%")