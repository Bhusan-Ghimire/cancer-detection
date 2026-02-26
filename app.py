import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from fastapi.middleware.cors import CORSMiddleware

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in this setup
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model once
MODEL_PATH = os.getenv("MODEL_PATH", "skin_cancer_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)

    prob = model.predict(img)[0][0]
    label = "Cancerous" if prob >= 0.6 else "Normal"

    return {
        "prediction": label,
        "confidence": float(prob)
    }
