const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const PORT = 3000;
const FASTAPI_URL = "http://127.0.0.1:8000";

// Enable CORS for React frontend
app.use(cors());

// Multer stores uploaded files in memory
const upload = multer({ storage: multer.memoryStorage() });

// Health check
app.get("/", (req, res) => {
    res.json({ status: "Middleware is running" });
});

// Proxy /predict to FastAPI
app.post("/predict", upload.single("file"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        // Build a new FormData to forward to FastAPI
        const form = new FormData();
        form.append("file", req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype,
        });

        const response = await axios.post(`${FASTAPI_URL}/predict`, form, {
            headers: form.getHeaders(),
        });

        res.json(response.data);
    } catch (error) {
        console.error("Prediction error:", error.message);
        res.status(500).json({
            error: "Failed to get prediction",
            details: error.message,
        });
    }
});

app.listen(PORT, () => {
    console.log(`Middleware server running on http://localhost:${PORT}`);
});
