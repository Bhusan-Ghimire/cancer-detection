import { useState, useRef, useEffect } from "react";
import { predictImage } from "./api";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768 && /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent));
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

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

  const getRiskLevel = (confidence) => {
    const prob = confidence * 100;
    if (prob <= 30) return {
      label: "Low Risk",
      type: "safe",
      note: "No signs of malignancy",
      color: "safe"
    };
    if (prob <= 70) return {
      label: "Moderate Risk",
      type: "warning",
      note: "Further examination suggested",
      color: "warning"
    };
    return {
      label: "High Risk",
      type: "danger",
      note: "Suspicious lesion detected",
      color: "danger"
    };
  };

  const risk = result ? getRiskLevel(result.confidence) : null;

  return (
    <div className="app">
      {/* Background decorations */}
      <div className="bg-glow bg-glow-1"></div>
      <div className="bg-glow bg-glow-2"></div>
      <div className="bg-glow bg-glow-3"></div>

      {/* Floating medical cross icon */}
      <div className="floating-cross">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z" />
        </svg>
      </div>

      <header className="header">
        <div className="logo">
          <div className="logo-icon">
            {/* Shield logo with medical cross */}
            <svg width="44" height="44" viewBox="0 0 48 48" fill="none">
              <defs>
                <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#3182ce" />
                  <stop offset="100%" stopColor="#2b6cb0" />
                </linearGradient>
              </defs>
              {/* Shield shape */}
              <path
                d="M24 4L6 12v10c0 11.1 7.7 21.5 18 24 10.3-2.5 18-12.9 18-24V12L24 4z"
                fill="url(#shieldGrad)"
              />
              {/* Medical cross */}
              <rect x="20" y="14" width="8" height="20" rx="2" fill="white" />
              <rect x="14" y="20" width="20" height="8" rx="2" fill="white" />
            </svg>
          </div>
          <span>DermAI Scanner</span>
        </div>
        <p className="subtitle">AI-Powered Skin Lesion Analysis</p>
      </header>

      <main className="main">
        {/* Upload Section */}
        <section className="upload-section glass-card">
          <h2 className="section-title">{isMobile ? "Click Skin Image" : "Upload Skin Image"}</h2>
          <p className="section-desc">
            {isMobile
              ? "Click to capture a dermoscopic image for analysis"
              : "Drag & drop or click to upload a dermoscopic image for analysis"}
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
                <p className="dropzone-text">{isMobile ? "Click to take a photo" : "Drop your image here"}</p>
                <p className="dropzone-hint">
                  {isMobile ? "or select from gallery · JPG, PNG supported" : "or click to browse · JPG, PNG supported"}
                </p>
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
        {result && risk && (
          <section className="result-section glass-card fade-in">
            <h2 className="section-title">Analysis Result</h2>
            <div className={`result-badge result-${risk.type}`}>
              <div className="result-icon">
                {risk.type === "danger" ? (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                ) : risk.type === "warning" ? (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                ) : (
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                )}
              </div>
              <div className="result-info">
                <span className="result-label">{risk.label}</span>
                <span className="result-sublabel">{risk.note}</span>
              </div>
            </div>

            <div className="confidence-section">
              <div className="confidence-header">
                <span>Malignancy likelihood</span>
                <span className="confidence-value">{confidencePercent}%</span>
              </div>
              <div className="confidence-bar-track">
                <div
                  className={`confidence-bar-fill bar-${risk.color}`}
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
    </div>
  );
}

export default App;
