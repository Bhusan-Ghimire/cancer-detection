import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Load trained model once
MODEL_PATH = os.getenv("MODEL_PATH", "skin_cancer_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)

    prob = model.predict(img)[0][0]

    return {
        "confidence": float(prob)
    }
