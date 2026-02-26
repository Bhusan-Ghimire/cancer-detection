# Skin Cancer Detection — Full Stack Walkthrough

## Architecture

```mermaid
    A["React Frontend<br/>:5173"] -->|POST /predict| B["Express Middleware<br/>:3000"]
    B -->|POST /predict| C["FastAPI Server<br/>:8000"]
    C -->|TensorFlow| D["skin_cancer_model.keras"]
    D -->|JSON| C
    C -->|JSON| B
    B -->|JSON| A
    style A stroke:#6366f1,stroke-width:2px
    style B stroke:#6366f1,stroke-width:2px
    style C stroke:#6366f1,stroke-width:2px
```

*Note: Access wirelessly on your phone via `http://192.168.1.72:5173`*

## Files Created / Modified

| File | Action | Purpose |
|------|--------|---------|
| [app.py](file:///d:/ML%20app%20for%20publishing/cancer%20detection/app.py) | Modified | Fixed model path to [skin_cancer_model.keras](file:///d:/ML%20app%20for%20publishing/cancer%20detection/skin_cancer_model.keras), added CORS |
| [middleware/index.js](file:///d:/ML%20app%20for%20publishing/cancer%20detection/middleware/index.js) | New | Express proxy — receives image, forwards to FastAPI |
| [frontend/src/App.jsx](file:///d:/ML%20app%20for%20publishing/cancer%20detection/frontend/src/App.jsx) | New | Main React UI with drag-and-drop upload and result display |
| [frontend/src/App.css](file:///d:/ML%20app%20for%20publishing/cancer%20detection/frontend/src/App.css) | New | Premium dark theme with glassmorphism and animations |
| [frontend/src/api.js](file:///d:/ML%20app%20for%20publishing/cancer%20detection/frontend/src/api.js) | New | Axios service calling middleware `/predict` |
| [frontend/src/index.css](file:///d:/ML%20app%20for%20publishing/cancer%20detection/frontend/src/index.css) | Modified | Inter font import and base styles |

## How to Run

Start all three servers in separate terminals:

```bash
# Terminal 1 — FastAPI (model server)
uvicorn app:app --reload --port 8000

# Terminal 2 — Express middleware
cd middleware && node index.js

# Terminal 3 — React frontend
cd frontend && npm run dev
```

Then open **http://192.168.1.72:5173** in your browser (on PC or phone).

## Verification Results

| Server | URL | Status |
|--------|-----|--------|
| FastAPI | http://localhost:8000 | ✅ Running |
| Express Middleware | http://192.168.1.72:3000 | ✅ Running |
| React Frontend | http://192.168.1.72:5173 | ✅ Running |

## Response Format

The `/predict` endpoint returns:

```json
{
  "prediction": "Cancerous" | "Normal",
  "confidence": 0.85
}
```
