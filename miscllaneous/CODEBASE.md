# DermAI Scanner — Complete Codebase Reference

> This document is the single source of truth for the entire project. It describes every file, the architecture, the ML pipeline,
> the full-stack request flow, all design decisions, and how to run the app. An AI model reading this file should understand
> the project in its entirety without needing to open any other file.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Architecture & Request Flow](#3-architecture--request-flow)
4. [Layer 1 — ML Training Script (`model.py`)](#4-layer-1--ml-training-script-modelpy)
5. [Layer 2 — FastAPI Server (`app.py`)](#5-layer-2--fastapi-server-apppy)
6. [Layer 3 — Express Middleware (`middleware/index.js`)](#6-layer-3--express-middleware-middlewareindexjs)
7. [Layer 4 — React Frontend (`frontend/`)](#7-layer-4--react-frontend-frontend)
   - [Entry Point](#71-entry-point-indexhtml--mainjsx)
   - [API Service (`api.js`)](#72-api-service-apijs)
   - [Main Component (`App.jsx`)](#73-main-component-appjsx)
   - [Styling (`App.css` & `index.css`)](#74-styling-appcss--indexcss)
8. [Dataset & Data Files](#8-dataset--data-files)
9. [Dependencies Summary](#9-dependencies-summary)
10. [Running the App](#10-running-the-app)
11. [Key ML Concepts & Design Decisions](#11-key-ml-concepts--design-decisions)
12. [Miscellaneous / Development Notes](#12-miscellaneous--development-notes)

---

## 1. Project Overview

**DermAI Scanner** is a full-stack skin cancer detection web application. A user uploads a dermoscopic image of a skin lesion
through a React frontend. The image travels through an Express.js middleware proxy, then to a FastAPI server powered by a
TensorFlow/Keras deep learning model. The model classifies the lesion as either **"Cancerous"** or **"Normal"** and returns a
confidence score. The result is displayed instantly in the UI.

**Tech Stack:**
| Layer | Technology |
|---|---|
| Machine Learning | Python, TensorFlow/Keras, MobileNetV2 (transfer learning) |
| Model Server | Python, FastAPI, Uvicorn |
| Middleware | Node.js, Express.js |
| Frontend | React 19 (Vite), Vanilla CSS |
| HTTP Client | Axios (used on both frontend and middleware) |

---

## 2. Directory Structure

```
cancer detection/                      ← project root
│
├── model.py                           ← ML training script (offline, run once)
├── app.py                             ← FastAPI server (serves the trained model)
├── skin_cancer_model.keras            ← PRODUCTION trained model (used by app.py)
├── skin_cancer_model_testing.keras    ← Experimental / latest training output
├── metadata.csv                       ← ISIC dataset metadata (5.3 MB, ~50k rows)
│
├── ISIC-images/                       ← Dermoscopic JPEG images from the ISIC dataset
│                                         (not tracked in git due to size)
│
├── middleware/
│   ├── index.js                       ← Express proxy server (Port 3000)
│   ├── package.json                   ← Node dependencies: express, cors, multer, axios, form-data
│   └── node_modules/
│
├── frontend/
│   ├── index.html                     ← Root HTML shell (mounts React app)
│   ├── vite.config.js                 ← Vite build config (React plugin)
│   ├── package.json                   ← React 19, Axios, Vite, ESLint
│   ├── eslint.config.js
│   └── src/
│       ├── main.jsx                   ← React entry point (renders <App /> into #root)
│       ├── App.jsx                    ← Entire frontend application UI logic
│       ├── App.css                    ← All component styles (dark glassmorphism theme)
│       ├── index.css                  ← Global reset + Inter font import
│       ├── api.js                     ← Axios call to middleware /predict endpoint
│       └── assets/                    ← Static assets (Vite default)
│
└── miscllaneous/
    ├── v1.py                          ← First version of the model (CNN from scratch, archived)
    ├── some_explanations.txt          ← Developer notes on TF/Keras concepts used
    └── walkthrough.md                 ← Architecture diagram + run instructions (quick ref)
```

---

## 3. Architecture & Request Flow

```
User (Browser)
    │
    │  drag & drop / file select (image file)
    ▼
React Frontend  ─── http://localhost:5173  (Vite dev server)
    │
    │  POST /predict  multipart/form-data  (via Axios)
    ▼
Express Middleware  ─── http://localhost:3000
    │  - Receives file with multer (stored in memory)
    │  - Re-packs into new FormData
    │  - Forwards to FastAPI via Axios
    ▼
FastAPI Server  ─── http://localhost:8000
    │  - Receives UploadFile
    │  - Preprocesses image (resize to 224x224, normalize to [0,1])
    │  - Runs model.predict()
    │  - Applies threshold: prob >= 0.6 → "Cancerous", else → "Normal"
    ▼
TensorFlow Model (skin_cancer_model.keras)
    │  MobileNetV2 base + custom Conv/Dense head
    │  Outputs sigmoid probability [0.0 – 1.0]
    ▼
FastAPI  →  Express  →  React
    └── JSON: { "prediction": "Cancerous"|"Normal", "confidence": 0.85 }
```

**Why three servers?**
- **FastAPI** is needed because the TensorFlow model runs in Python. FastAPI is the standard, fast Python web framework for ML inference.
- **Express middleware** acts as a proxy/adapter between the browser and FastAPI. It handles CORS cleanly, re-packages the file upload,
  and decouples the frontend from knowing anything about the Python backend's specifics. This is a common pattern when combining
  a JavaScript frontend with a Python ML backend.
- **React frontend** runs separately during development (Vite dev server) and communicates with the Express proxy.

---

## 4. Layer 1 — ML Training Script (`model.py`)

**Purpose:** Train the skin cancer classification model. This script is run **offline once** to produce `skin_cancer_model.keras`.
It is NOT part of the live serving stack.

### Configuration (top of file)
```python
IMAGE_DIR = "./ISIC-images"    # directory of JPEG images
CSV_PATH  = "./metadata.csv"   # ISIC metadata CSV with labels
IMG_SIZE  = (224, 224)         # MobileNetV2 required input size
BATCH_SIZE = 8
EPOCHS     = 300               # early stopping will cut this short
RANDOM_STATE = 42              # reproducibility seed
TOTAL_IMAGES = 1000            # number of images to use (subset of full dataset)
SKIP_IMAGES  = 1000            # skip first N images to introduce variety
```

### Data Loading & Label Extraction
```python
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=RANDOM_STATE)   # shuffle reproducibly
df = df.iloc[SKIP_IMAGES : SKIP_IMAGES + TOTAL_IMAGES]

image_paths = [os.path.join(IMAGE_DIR, isic_id + ".jpg") for isic_id in df["isic_id"]]
labels = df["melanocytic"].astype(int).values   # 1 = melanocytic (cancerous), 0 = normal
```
- Labels are drawn from the `melanocytic` column of `metadata.csv`.
- `melanocytic = 1` indicates a melanocytic lesion (potentially malignant / cancerous).
- `melanocytic = 0` indicates a non-melanocytic lesion (benign / normal).

### TF Data Pipeline
```python
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)  # preserves aspect ratio
    img = tf.cast(img, tf.float32) / 255.0          # normalize to [0, 1]
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(512, seed=RANDOM_STATE)
```

**Split:** 70% train / 15% validation / 15% test
```python
train_ds = dataset.take(700).batch(8).prefetch(tf.data.AUTOTUNE)
val_ds   = dataset.skip(700).take(150).batch(8).prefetch(tf.data.AUTOTUNE)
test_ds  = dataset.skip(850).batch(8)
```
- `prefetch(AUTOTUNE)` allows CPU to prepare the next batch while GPU processes the current one.

### Model Architecture — Transfer Learning (MobileNetV2)
The final deployed model is `transfer_model`, which builds on top of MobileNetV2:

```python
data_augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom((-0.2, 0.1))
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze pretrained weights

transfer_model = models.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    data_augmentation_layer,           # augmentation during training only
    base_model,                        # MobileNetV2 feature extractor
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(negative_slope=0.1),
    layers.GlobalAveragePooling2D(),   # spatial → single vector
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # binary output
])
```

**Why MobileNetV2?**
- Pre-trained on ImageNet (general visual features transfer well to medical imaging).
- Lightweight — suitable for a smaller dataset (1000 images) without overfitting risk.
- `include_top=False` removes ImageNet classification head; we add our own.
- Frozen base means we only train our custom head layers.

**There is also a `model_from_scratch` (CNN built from scratch) defined in the file:**
```
Input(224,224,3) → Conv2D(32) → BN → LeakyReLU
               → Conv2D(64)  → BN → LeakyReLU → MaxPool
               → Conv2D(128) → BN → LeakyReLU → MaxPool
               → Conv2D(128) → BN → LeakyReLU → MaxPool
               → GlobalAveragePooling2D
               → Dense(256, l2=1e-3) → LeakyReLU → Dropout(0.5)
               → Dense(1, sigmoid)
```
This is **not** used in production. `model = transfer_model` is the selected model.

### Compilation
```python
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=BinaryCrossentropy(),
    metrics=['recall']    # recall is prioritized over accuracy for cancer detection
)
```
**Why recall?** In cancer detection, a false negative (missing a cancer) is far more harmful than a false positive.
Monitoring recall ensures the model is catching as many cancer cases as possible.

### Class Imbalance Handling
```python
weights = compute_class_weight("balanced", classes=[0,1], y=labels)
priority_factor = 1.15
class_weight = {0: weights[0], 1: weights[1] * priority_factor}
```
- sklearn computes the inverse-frequency weights.
- An extra `priority_factor = 1.15` further penalizes missing cancerous cases.
- These weights are passed into `model.fit(class_weight=class_weight)`.

### Training
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(train_ds, validation_data=val_ds, epochs=300,
                    class_weight=class_weight, callbacks=[early_stopping])
model.save("skin_cancer_model_testing.keras")
```

### Evaluation & Inference
After training, the script runs a confusion matrix on the test set and provides a `predict_single_image()` utility:
```python
def predict_single_image(image_path):
    # Load → decode → resize_with_pad → normalize → add batch dim
    # model.predict() → sigmoid value → threshold 0.5 → "Cancerous" / "Normal"
```
Note: Training threshold is 0.5 but the live serving threshold in `app.py` is **0.6** (more conservative for production).

---

## 5. Layer 2 — FastAPI Server (`app.py`)

**Purpose:** Load the trained model and expose a `/predict` HTTP endpoint.
Runs on **port 8000** with `uvicorn app:app --reload`.

```python
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS: allow all origins (development setup, restrict in production)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Model loaded once at startup — not per request
model = tf.keras.models.load_model("skin_cancer_model.keras")

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # handles PNG/JPEG/etc
    img = img.resize(IMG_SIZE)        # no aspect-ratio preservation here (PIL resize)
    img = np.array(img) / 255.0      # normalize
    img = np.expand_dims(img, axis=0) # add batch dimension: (1, 224, 224, 3)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    prob = model.predict(img)[0][0]                        # scalar sigmoid output
    label = "Cancerous" if prob >= 0.6 else "Normal"      # threshold = 0.6
    return {"prediction": label, "confidence": float(prob)}
```

**Key points:**
- The model is loaded **once** when the server starts (not per request) — essential for performance.
- `PIL.Image` is used for preprocessing here (vs TensorFlow ops in training). Result is equivalent.
- **Threshold is 0.6** (higher than training's 0.5) — intentionally conservative in production to reduce
  false negatives further while not overwhelming users with false positives.
- Only one route: `POST /predict` accepts a file upload and returns JSON.

---

## 6. Layer 3 — Express Middleware (`middleware/index.js`)

**Purpose:** Bridge the React frontend and the FastAPI server. Runs on **port 3000**.

```javascript
const express = require("express");
const cors    = require("cors");
const multer  = require("multer");
const axios   = require("axios");
const FormData = require("form-data");

const app = express();
const PORT = 3000;
const FASTAPI_URL = "http://127.0.0.1:8000";

app.use(cors());                                         // allow React (5173) to call this
const upload = multer({ storage: multer.memoryStorage() }); // file lives in RAM, not disk

// Health check
app.get("/", (req, res) => res.json({ status: "Middleware is running" }));

// Main proxy route
app.post("/predict", upload.single("file"), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const form = new FormData();
    form.append("file", req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
    });

    const response = await axios.post(`${FASTAPI_URL}/predict`, form, {
        headers: form.getHeaders(),
    });

    res.json(response.data);  // transparently forwards FastAPI's JSON response
});

app.listen(PORT, () => console.log(`Middleware server running on http://localhost:${PORT}`));
```

**Key points:**
- `multer.memoryStorage()` — the uploaded file is kept in `req.file.buffer` (RAM), never written to disk.
- The middleware **re-creates** a new `FormData` to forward to FastAPI, because Axios requires this pattern
  to properly set `multipart/form-data` headers when proxying binary data.
- `form.getHeaders()` sets the correct `Content-Type: multipart/form-data; boundary=...` header automatically.
- All errors are caught and returned as JSON with a 500 status.

**Node dependencies (`middleware/package.json`):**
| Package | Version | Purpose |
|---|---|---|
| express | ^5.2.1 | HTTP server framework |
| cors | ^2.8.6 | Cross-Origin Resource Sharing middleware |
| multer | ^2.0.2 | Multipart file upload parsing |
| axios | ^1.13.5 | HTTP client to call FastAPI |
| form-data | ^4.0.5 | Build multipart FormData in Node.js |

---

## 7. Layer 4 — React Frontend (`frontend/`)

Built with **React 19 + Vite**. Runs on **port 5173** during development.

### 7.1 Entry Point (`index.html` + `main.jsx`)

`frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>DermAI Scanner</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

`frontend/src/main.jsx`:
```jsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
)
```
Mounts `<App />` into the `#root` div. `StrictMode` enables extra React warnings in development.

---

### 7.2 API Service (`api.js`)

```javascript
import axios from "axios";

const MIDDLEWARE_URL = "http://localhost:3000";

export async function predictImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(`${MIDDLEWARE_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });

    return response.data;  // { prediction: "...", confidence: 0.xx }
}
```
This is the **only** network call in the frontend. The function is a thin wrapper around Axios, kept separate
from the component so the component doesn't care about HTTP details.

---

### 7.3 Main Component (`App.jsx`)

The entire frontend UI lives in a single component. Here is a breakdown of its state and logic:

**State:**
| State variable | Type | Purpose |
|---|---|---|
| `file` | File \| null | The selected image file object |
| `preview` | string \| null | Object URL for displaying the selected image |
| `result` | object \| null | `{ prediction, confidence }` from the API |
| `loading` | boolean | Shows spinner while API call is in flight |
| `error` | string \| null | Error message if API call fails |
| `dragActive` | boolean | Tracks whether a file is being dragged over the dropzone |

**Ref:**
- `inputRef` — reference to the hidden `<input type="file">` so the visible dropzone div can trigger it.

**Event Handlers:**
```
handleFile(selectedFile)  — validates file is an image, sets file + preview, clears result/error
handleDrop(e)             — handles drop event, extracts file, calls handleFile()
handleDrag(e)             — manages dragenter/dragover/dragleave to toggle dragActive style
handleSubmit()            — calls predictImage(file), sets result on success, error on failure
handleReset()             — clears all state back to initial
```

**Render structure (JSX):**
```
<div class="app">
  <div class="bg-glow bg-glow-1/2/3" />   ← decorative animated background blobs

  <header class="header">
    <div class="logo">                     ← SVG icon + "DermAI Scanner" text
    <p class="subtitle">                   ← "AI-Powered Skin Lesion Analysis"

  <main class="main">
    <section class="upload-section glass-card">
      <h2>Upload Skin Image</h2>
      <div class="dropzone" [drag events] [click→inputRef]>
        <input type="file" hidden ref={inputRef} />
        if (preview):   <img preview-image> with hover overlay "Click to change image"
        else:           upload icon SVG + hint text
      </div>
      <div class="actions">
        if (file && !loading):  <button "Analyze Image"> + <button "Reset">
        if (loading):           <div class="loading"> spinner + "Analyzing..."

    if (result):
      <section class="result-section glass-card fade-in">
        <div class="result-badge result-danger|result-safe">
          ← warning/checkmark SVG icon + prediction label + sub-label
        <div class="confidence-section">
          ← progress bar animated to confidence%
        <div class="disclaimer">
          ← "Always consult a dermatologist" notice

    if (error):
      <section class="error-section glass-card">
        ← error message text

  <footer>DermAI Scanner · Powered by TensorFlow & React
```

**Result display logic:**
- `result.prediction === "Cancerous"` → red badge (`result-danger`), warning triangle SVG, red progress bar
- otherwise → green badge (`result-safe`), checkmark SVG, green progress bar
- `confidencePercent = (result.confidence * 100).toFixed(1)` — converts 0.85 → "85.0"

---

### 7.4 Styling (`App.css` & `index.css`)

The app uses a **premium dark glassmorphism theme**. All styles are Vanilla CSS with CSS custom properties.

**`index.css`** — global base:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root { font-family: 'Inter', sans-serif; line-height: 1.6; }
/* antialiasing and legibility settings */
```

**`App.css`** — design tokens and all component styles:

**CSS Custom Properties (design tokens):**
```css
:root {
  --bg-primary:    #0a0e1a;          /* near-black background */
  --bg-secondary:  #111827;
  --surface:       rgba(255,255,255,0.04);  /* glass card background */
  --border:        rgba(255,255,255,0.08);
  --border-active: rgba(99,102,241,0.5);   /* indigo when active */
  --text-primary:  #f1f5f9;
  --text-secondary:#94a3b8;
  --text-muted:    #64748b;
  --accent:        #6366f1;          /* indigo — primary brand color */
  --accent-glow:   rgba(99,102,241,0.3);
  --danger:        #ef4444;          /* red — for cancerous result */
  --safe:          #10b981;          /* emerald — for normal result */
  --radius-xl:     24px;             /* card border radius */
}
```

**Animations defined:**
| Animation | CSS class | Behavior |
|---|---|---|
| `float` | `.bg-glow-1/2/3` | Background blobs slowly drift (20–25s infinite) |
| `slideDown` | `.header` | Header fades + slides down on load (0.6s) |
| `bob` | `.upload-icon` | Upload icon bobs up/down every 3s |
| `spin` | `.spinner` | Loading spinner rotates (0.8s linear) |
| `fadeUp` | `.result-section`, `.fade-in` | Results fade + slide up on appear (0.5s) |

**Glass card effect:**
```css
.glass-card {
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 24px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
```

**Responsive:** Media query at `max-width: 480px` adjusts padding, logo size, and stacks the result badge vertically.

**Frontend dependencies (`frontend/package.json`):**
| Package | Version | Purpose |
|---|---|---|
| react | ^19.2.0 | UI framework |
| react-dom | ^19.2.0 | React DOM renderer |
| axios | ^1.13.5 | HTTP client for API calls |
| vite | ^7.3.1 | Dev server + build tool |
| @vitejs/plugin-react | ^5.1.1 | Vite plugin for JSX/React Fast Refresh |

---

## 8. Dataset & Data Files

### `metadata.csv` (5.3 MB)
- Source: **ISIC (International Skin Imaging Collaboration)** Archive
- Each row represents one skin lesion image
- Key column used: **`isic_id`** (the filename, e.g. `ISIC_0024306`) and **`melanocytic`** (binary label)
- `melanocytic = 1`: melanocytic lesion (cancer-related cell type) → labeled **Cancerous**
- `melanocytic = 0`: non-melanocytic lesion → labeled **Normal**

### `ISIC-images/`
- Directory of `.jpg` dermoscopic photographs
- Filenames match `isic_id` values in `metadata.csv`
- Not tracked in git (too large)

### Model Files
| File | Size | Description |
|---|---|---|
| `skin_cancer_model.keras` | ~27 MB | Production model — used by `app.py` |
| `skin_cancer_model_testing.keras` | ~27 MB | Output of the most recent `model.py` run |

Both are TensorFlow SavedModel format wrapped in Keras v3 `.keras` format.

---

## 9. Dependencies Summary

### Python (`app.py` and `model.py`)
```
tensorflow         — model training (Keras API) + inference
fastapi            — HTTP server framework
uvicorn            — ASGI server to run FastAPI
pillow (PIL)       — image loading/resizing in app.py
numpy              — array operations
pandas             — CSV reading and metadata manipulation
scikit-learn       — class weight computation, confusion matrix display
matplotlib         — plotting confusion matrix (offline, during training only)
python-multipart   — required by FastAPI to handle multipart file uploads
```

### Node.js — Middleware
```
express            — HTTP server
cors               — CORS headers
multer             — parse multipart/form-data uploads (in-memory)
axios              — HTTP client to call FastAPI
form-data          — construct multipart form for forwarding
```

### Node.js — Frontend
```
react, react-dom   — UI framework
axios              — HTTP client
vite               — dev server and bundler
@vitejs/plugin-react — JSX + Fast Refresh
```

---

## 10. Running the App

Three terminals are required simultaneously:

### Terminal 1 — FastAPI Model Server
```bash
# From project root: d:\ML app for publishing\cancer detection\
uvicorn app:app --reload --port 8000
```
- Starts the Python server at http://localhost:8000
- `--reload` watches for code changes and restarts automatically

### Terminal 2 — Express Middleware
```bash
cd middleware
node index.js
```
- Starts proxy server at http://localhost:3000
- Confirm: browser visit to http://localhost:3000 returns `{"status":"Middleware is running"}`

### Terminal 3 — React Frontend
```bash
cd frontend
npm run dev
```
- Starts Vite dev server at http://localhost:5173
- Open this URL in the browser to use the app

### Order matters on startup
FastAPI should ideally be running first before the middleware is tested, but the frontend/middleware can
start in any order — they only communicate at request time.

---

## 11. Key ML Concepts & Design Decisions

### Why Transfer Learning?
The dataset is small (1000 images). Training a deep CNN from scratch on 700 training images would almost certainly
overfit. MobileNetV2 pre-trained on ImageNet has already learned general features (edges, textures, shapes) that
transfer well to dermoscopy images. We freeze the base model (don't update its 2.2M parameters) and only train
the small custom head we add on top.

### Data Augmentation (only during training)
```python
RandomFlip("horizontal_and_vertical")
RandomRotation(0.2)         # ±20 degrees
RandomZoom((-0.2, 0.1))     # slight zoom in/out
```
Augmentation is placed **inside the model** using Keras preprocessing layers. This means:
- It applies only during training (disabled during `model.predict()` automatically)
- It happens on-the-fly each epoch — each image looks slightly different each time
- Hue/brightness/contrast augmentation is deliberately avoided to prevent benign images
  from appearing malignant (documented in `some_explanations.txt`)

### BatchNormalization Order
Applied after `Conv2D` and before the activation function:
`Conv2D → BatchNormalization → LeakyReLU`
This follows the original Batch Normalization paper recommendation.

### LeakyReLU vs ReLU
`negative_slope=0.1` — allows small gradients for negative inputs, avoiding "dying ReLU" problem where neurons
with always-negative inputs receive zero gradient and stop learning.

### GlobalAveragePooling2D vs Flatten
`GlobalAveragePooling2D` reduces spatial output `(batch, H, W, C)` → `(batch, C)` by averaging each channel's
spatial values. Compared to Flatten, it:
- Produces far fewer parameters in following Dense layers
- Provides some translation invariance
- Helps prevent overfitting on small datasets

### Class Weight Balancing
The ISIC dataset has class imbalance (~68%+ melanocytic). Without balancing, the model could predict "Cancerous"
every time and still achieve high "accuracy". Class weights tell the optimizer to penalize misclassification of
the minority class more heavily. An extra `priority_factor = 1.15` further boosts the weight of the cancerous class.

### Prediction Thresholds
| Context | Threshold | Reason |
|---|---|---|
| Training evaluation | 0.5 | Standard binary threshold |
| Production (`app.py`) | 0.6 | More conservative — slightly higher bar to call something "Cancerous", reducing false positives while still maintaining sensitivity |

### tf.data Pipeline
- `from_tensor_slices()` — pairs file paths with labels lazily (no images loaded upfront)
- `map()` with `AUTOTUNE` — parallel image loading/decoding using as many CPU threads as TF chooses
- `shuffle(512)` — maintains a buffer of 512 images and samples randomly from it
- `batch()` — groups samples into batches of 8
- `prefetch(AUTOTUNE)` — prepares next batch while model trains on current batch (CPU/GPU overlap)

---

## 12. Miscellaneous / Development Notes

### `miscllaneous/v1.py` — First Version (Archived)
The original model was a CNN built entirely from scratch (no transfer learning), trained on 300 images.
It used `PIL` and `numpy` for loading (eager, all images loaded into RAM at once) and `train_test_split`
from sklearn. This approach was abandoned in favor of the tf.data pipeline + transfer learning in `model.py`.
The file is kept for reference.

### `miscllaneous/some_explanations.txt`
Developer notes written during the project explaining TensorFlow concepts:
- How `from_tensor_slices` works
- What `prefetch` and `num_parallel_calls=AUTOTUNE` do
- The purpose of `GlobalAveragePooling2D` vs Flatten
- BatchNormalization layer ordering
- Why hue/contrast augmentation was avoided

### `miscllaneous/walkthrough.md`
A quick architectural reference with a Mermaid diagram and the run commands. Was generated when the full-stack
connection was first set up. The `CODEBASE.md` file (this file) supersedes it.

### `.gitignore` Files
- Root: ignores `__pycache__/`, `.keras` model files, dataset images (large binary files)
- Frontend: ignores `node_modules/`, `dist/`
- Middleware: ignores `node_modules/`

### Naming Note
The folder `miscllaneous/` contains a typo (extra 'l') — this is intentional/as-is and was the original folder name.

---

*Last updated: 2026-02-26*
