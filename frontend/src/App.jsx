import { useState, useRef } from "react";
import { predictImage } from "./api";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith("image/")) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError("Prediction failed. Make sure backend servers are running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const confidencePercent = result
    ? (result.confidence * 100).toFixed(1)
    : null;

  return (
    <div className="app">
      {/* Background decorations */}
      <div className="bg-glow bg-glow-1"></div>
      <div className="bg-glow bg-glow-2"></div>
      <div className="bg-glow bg-glow-3"></div>

      <header className="header">
        <div className="logo">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <span>DermAI Scanner</span>
        </div>
        <p className="subtitle">AI-Powered Skin Lesion Analysis</p>
      </header>

      <main className="main">
        {/* Upload Section */}
        <section className="upload-section glass-card">
          <h2 className="section-title">Upload Skin Image</h2>
          <p className="section-desc">
            Drag & drop or click to upload a dermoscopic image for analysis
          </p>

          <div
            className={`dropzone ${dragActive ? "dropzone-active" : ""} ${preview ? "dropzone-has-image" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <div className="preview-overlay">
                  <span>Click to change image</span>
                </div>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="upload-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <p className="dropzone-text">Drop your image here</p>
                <p className="dropzone-hint">or click to browse · JPG, PNG supported</p>
              </div>
            )}
          </div>

          <div className="actions">
            {file && !loading && (
              <>
                <button className="btn btn-primary" onClick={handleSubmit}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <path d="M21 21l-4.35-4.35" />
                  </svg>
                  Analyze Image
                </button>
                <button className="btn btn-ghost" onClick={handleReset}>
                  Reset
                </button>
              </>
            )}
            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <span>Analyzing...</span>
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {result && (
          <section className="result-section glass-card fade-in">
            <h2 className="section-title">Analysis Result</h2>
            <div className={`result-badge ${result.prediction === "Cancerous" ? "result-danger" : "result-safe"}`}>
              <div className="result-icon">
                {result.prediction === "Cancerous" ? (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                ) : (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                )}
              </div>
              <div className="result-info">
                <span className="result-label">{result.prediction}</span>
                <span className="result-sublabel">
                  {result.prediction === "Cancerous"
                    ? "Suspicious lesion detected"
                    : "No signs of malignancy"}
                </span>
              </div>
            </div>

            <div className="confidence-section">
              <div className="confidence-header">
                <span>Confidence Score</span>
                <span className="confidence-value">{confidencePercent}%</span>
              </div>
              <div className="confidence-bar-track">
                <div
                  className={`confidence-bar-fill ${result.prediction === "Cancerous" ? "bar-danger" : "bar-safe"}`}
                  style={{ width: `${confidencePercent}%` }}
                ></div>
              </div>
            </div>

            <div className="disclaimer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <span>This is an AI-based screening tool. Always consult a dermatologist for clinical diagnosis.</span>
            </div>
          </section>
        )}

        {error && (
          <section className="error-section glass-card fade-in">
            <p className="error-text">{error}</p>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>DermAI Scanner · Powered by TensorFlow & React</p>
      </footer>
    </div>
  );
}

export default App;
