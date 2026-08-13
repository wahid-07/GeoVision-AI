"""
Land Cover Classification Flask API.

Provides the original single-image classifier plus a new heatmap analysis
endpoint with overlapping grids, confidence-aware refinement, and uncertainty
outputs.
"""

from datetime import datetime
from pathlib import Path
import os

from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from stages.cell_inference.core.landcover_inference_pipeline import (
    CANONICAL_CLASS_NAMES,
    get_device,
    load_model,
    predict_heatmap,
    predict_single_image,
    render_heatmap_outputs,
)


app = Flask(__name__)

MODEL_PATH = 'best_eurosat_model.pth'
MODEL = None
DEVICE = None


def _resolve_model_path() -> str:
    candidates = [
        Path(MODEL_PATH),
        Path.cwd() / MODEL_PATH,
        Path(__file__).resolve().parents[3] / MODEL_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    project_root = Path(__file__).resolve().parents[3]
    for checkpoint in sorted(project_root.rglob('*.pth')):
        name = checkpoint.name.lower()
        if 'eurosat' in name or 'landcover' in name or 'best' in name:
            return str(checkpoint)

    return str(Path(MODEL_PATH))

HTML_TEMPLATE = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GeoVision AI</title>
  <style>
    :root {
      --bg: #07111f;
      --bg-soft: #0d1b2d;
      --panel: rgba(14, 27, 45, 0.82);
      --panel-strong: #10233e;
      --surface: rgba(19, 35, 58, 0.9);
      --line: rgba(148, 163, 184, 0.2);
      --text: #e5eefb;
      --muted: #9eb6d6;
      --primary: #5eead4;
      --primary-2: #7dd3fc;
      --secondary: #a78bfa;
      --warning: #fbbf24;
      --danger: #f87171;
      --success: #34d399;
      --shadow: 0 28px 60px rgba(2, 6, 23, 0.45);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(94, 234, 212, 0.15), transparent 28%),
        radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.14), transparent 30%),
        linear-gradient(135deg, var(--bg) 0%, #091827 100%);
      color: var(--text);
      min-height: 100vh;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding: 12px 18px;
      background: rgba(9, 20, 35, 0.7);
      border: 1px solid var(--line);
      border-radius: 18px;
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.8rem;
      color: var(--muted);
    }

    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      display: grid;
      place-items: center;
      color: #03131d;
      font-weight: 900;
      box-shadow: 0 12px 24px rgba(94, 234, 212, 0.35);
    }

    .status-pill {
      padding: 10px 16px;
      border-radius: 999px;
      border: 1px solid rgba(52, 211, 153, 0.45);
      background: rgba(16, 185, 129, 0.1);
      color: var(--success);
      font-size: 0.8rem;
      font-weight: 700;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 24px;
      margin-bottom: 28px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 24px;
      backdrop-filter: blur(18px);
    }

    .hero-copy {
      background:
        linear-gradient(135deg, rgba(17, 38, 58, 0.9), rgba(13, 27, 45, 0.72));
    }

    h1 {
      margin: 0 0 16px;
      font-size: clamp(2.2rem, 4vw, 3.6rem);
      line-height: 1.05;
      letter-spacing: -0.06em;
    }

    .accent {
      background: linear-gradient(135deg, var(--primary), var(--primary-2), var(--secondary));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .lead {
      margin: 0;
      max-width: 620px;
      line-height: 1.7;
      color: var(--muted);
      font-size: 1.05rem;
    }

    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 22px;
    }

    .stat {
      min-width: 120px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
    }

    .stat-label {
      display: block;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .stat-value {
      font-weight: 800;
      font-size: 1.2rem;
    }

    .mini-panel {
      display: flex;
      flex-direction: column;
      gap: 18px;
      justify-content: center;
    }

    .mini-title {
      margin: 0;
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .class-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .badge {
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(125, 211, 252, 0.06);
      color: var(--text);
      font-size: 0.8rem;
    }

    .workspace {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 24px;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      gap: 12px;
    }

    .panel-title {
      margin: 0;
      font-size: 1.15rem;
      letter-spacing: -0.03em;
    }

    .mode-toggle {
      display: inline-flex;
      background: rgba(148, 163, 184, 0.08);
      padding: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      gap: 6px;
    }

    .mode-toggle button {
      background: transparent;
      border: none;
      color: var(--muted);
      padding: 8px 10px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 700;
    }

    .mode-toggle button.active {
      background: linear-gradient(135deg, rgba(94, 234, 212, 0.18), rgba(125, 211, 252, 0.12));
      color: var(--text);
      border: 1px solid rgba(94, 234, 212, 0.4);
    }

    .upload-box {
      border: 1.5px dashed rgba(148, 163, 184, 0.4);
      border-radius: 22px;
      padding: 22px;
      background: rgba(148, 163, 184, 0.02);
      transition: 0.2s ease;
    }

    .upload-box.dragover {
      border-color: var(--primary);
      background: rgba(94, 234, 212, 0.04);
    }

    .hidden-input {
      display: none;
    }

    .file-label {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      width: 100%;
      min-height: 90px;
      padding: 16px;
      border-radius: 16px;
      background: rgba(94, 234, 212, 0.06);
      border: 1px solid rgba(94, 234, 212, 0.18);
      color: var(--text);
      cursor: pointer;
      font-weight: 700;
      text-align: center;
    }

    .file-feedback {
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .field label {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 600;
    }

    .field input, .field select {
      appearance: none;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--line);
      border-radius: 12px;
      color: var(--text);
      padding: 11px 12px;
      outline: none;
      font-size: 0.96rem;
    }

    .field input:focus, .field select:focus {
      border-color: rgba(94, 234, 212, 0.7);
      box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.12);
    }

    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 12px;
    }

    .checkbox-row input {
      accent-color: var(--primary);
      width: 16px;
      height: 16px;
    }

    .cta-row {
      display: flex;
      gap: 12px;
      margin-top: 22px;
      flex-wrap: wrap;
    }

    .button {
      border: none;
      border-radius: 14px;
      padding: 12px 18px;
      font-weight: 800;
      cursor: pointer;
      transition: 0.2s ease;
    }

    .button.primary {
      background: linear-gradient(135deg, var(--primary), var(--primary-2));
      color: #04131d;
      box-shadow: 0 18px 28px rgba(94, 234, 212, 0.2);
    }

    .button.secondary {
      background: rgba(148, 163, 184, 0.08);
      color: var(--text);
      border: 1px solid var(--line);
    }

    .button:hover {
      transform: translateY(-1px);
    }

    .result-panel {
      display: grid;
      gap: 18px;
    }

    .result-card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
    }

    .preview {
      width: 100%;
      min-height: 220px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(148,163,184,0.07), rgba(94,234,212,0.02)),
        linear-gradient(45deg, rgba(255,255,255,0.03) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.03) 75%),
        linear-gradient(45deg, rgba(255,255,255,0.03) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.03) 75%);
      background-size: auto, 24px 24px, 24px 24px;
      background-position: 0 0, 0 0, 12px 12px;
      display: grid;
      place-items: center;
      color: var(--muted);
      overflow: hidden;
    }

    .preview img {
      width: 100%;
      max-height: 340px;
      object-fit: contain;
      display: none;
      border-radius: 18px;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }

    .meta-box {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }

    .meta-box small {
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .meta-box strong {
      font-size: 1.05rem;
    }

    .prediction-list {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }

    .prediction-item {
      display: grid;
      grid-template-columns: 1.2fr 1fr auto;
      align-items: center;
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(148,163,184,0.12);
    }

    .prediction-item:last-child {
      border-bottom: none;
    }

    .bar {
      background: rgba(148,163,184,0.1);
      border-radius: 999px;
      overflow: hidden;
      height: 10px;
      position: relative;
    }

    .bar > span {
      display: block;
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(135deg, var(--primary), var(--secondary));
    }

    .status-text {
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 12px;
    }

    .error-box {
      display: none;
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(248, 113, 113, 0.08);
      border: 1px solid rgba(248, 113, 113, 0.28);
      color: #fecaca;
    }

    .hidden {
      display: none !important;
    }

    @media (max-width: 940px) {
      .hero, .workspace {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 560px) {
      .grid-2, .meta-grid {
        grid-template-columns: 1fr;
      }
      .topbar {
        flex-direction: column;
        align-items: flex-start;
      }
      .stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">G</div>
        <span>GeoVision AI</span>
      </div>
      <div class="status-pill">System active</div>
    </header>

    <section class="hero">
      <div class="card hero-copy">
        <h1>Land cover intelligence for <span class="accent">satellite imagery</span></h1>
        <p class="lead">
          Analyze aerial scenes with a trained EuroSAT-driven classifier, surface the top land-cover predictions,
          and inspect heatmap detail for uncertain regions with cell-level insights.
        </p>
        <div class="stats">
          <div class="stat">
            <span class="stat-label">Classes</span>
            <span class="stat-value">6</span>
          </div>
          <div class="stat">
            <span class="stat-label">Mode</span>
            <span class="stat-value">Dual</span>
          </div>
          <div class="stat">
            <span class="stat-label">API</span>
            <span class="stat-value">Flask</span>
          </div>
        </div>
      </div>

      <aside class="card mini-panel">
        <div>
          <h2 class="mini-title">Canonical taxonomy</h2>
          <div class="class-badges">
            {% for class_name in class_names %}
            <span class="badge">{{ class_name }}</span>
            {% endfor %}
          </div>
        </div>
        <div>
          <h2 class="mini-title">Endpoints</h2>
          <div class="class-badges">
            <span class="badge">/classify</span>
            <span class="badge">/analyze</span>
          </div>
        </div>
      </aside>
    </section>

    <main class="workspace">
      <section class="card">
        <div class="panel-header">
          <h2 class="panel-title">Image analysis</h2>
          <div class="mode-toggle" aria-label="Analysis mode selector">
            <button class="active" type="button" data-mode="single">Single</button>
            <button type="button" data-mode="heatmap">Heatmap</button>
          </div>
        </div>

        <div class="upload-box" id="uploadBox">
          <input id="imageInput" class="hidden-input" type="file" accept="image/*" />
          <label class="file-label" for="imageInput" id="fileLabel">
            Choose image or drop it here
          </label>
          <div class="file-feedback" id="fileFeedback">No file selected</div>
        </div>

        <div class="grid-2">
          <div class="field">
            <label for="cellSize">Cell size</label>
            <input id="cellSize" type="number" value="128" min="32" step="8" />
          </div>
          <div class="field">
            <label for="overlap">Overlap</label>
            <input id="overlap" type="number" value="0.25" min="0" max="0.75" step="0.05" />
          </div>
          <div class="field">
            <label for="temperature">Temperature</label>
            <input id="temperature" type="number" value="1.0" min="0.25" max="3" step="0.05" />
          </div>
          <div class="field">
            <label for="confidenceThreshold">Low-confidence threshold</label>
            <input id="confidenceThreshold" type="number" value="0.55" min="0.1" max="1" step="0.05" />
          </div>
        </div>

        <label class="checkbox-row">
          <input id="ttaToggle" type="checkbox" checked />
          Use test-time augmentation
        </label>

        <div class="cta-row">
          <button id="analyzeButton" class="button primary" type="button">Analyze image</button>
          <button id="resetButton" class="button secondary" type="button">Reset</button>
        </div>

        <div class="error-box" id="errorBox"></div>
      </section>

      <aside class="card result-panel">
        <div class="result-card">
          <div class="preview" id="previewContainer">
            <img id="previewImage" alt="Uploaded preview" />
            <span id="previewPlaceholder">No image preview yet</span>
          </div>
        </div>

        <div class="result-card">
          <div class="panel-header">
            <h2 class="panel-title">Prediction summary</h2>
          </div>
          <div class="meta-grid">
            <div class="meta-box">
              <small>Predicted class</small>
              <strong id="predictedClass">—</strong>
            </div>
            <div class="meta-box">
              <small>Confidence</small>
              <strong id="confidenceValue">—</strong>
            </div>
            <div class="meta-box">
              <small>Low confidence</small>
              <strong id="lowConfidence">—</strong>
            </div>
            <div class="meta-box">
              <small>Grid cells</small>
              <strong id="gridCells">—</strong>
            </div>
          </div>
          <div class="prediction-list" id="predictionList"></div>
          <div class="status-text" id="statusText">Upload an image and run a prediction.</div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const modeButtons = document.querySelectorAll('[data-mode]');
    const imageInput = document.getElementById('imageInput');
    const fileFeedback = document.getElementById('fileFeedback');
    const previewImage = document.getElementById('previewImage');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const predictButton = document.getElementById('analyzeButton');
    const errorBox = document.getElementById('errorBox');
    const predictedClass = document.getElementById('predictedClass');
    const confidenceValue = document.getElementById('confidenceValue');
    const lowConfidence = document.getElementById('lowConfidence');
    const gridCells = document.getElementById('gridCells');
    const predictionList = document.getElementById('predictionList');
    const statusText = document.getElementById('statusText');
    const uploadBox = document.getElementById('uploadBox');

    let currentMode = 'single';
    let selectedFile = null;

    function setMode(mode) {
      currentMode = mode;
      modeButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.mode === mode);
      });
    }

    function setError(message) {
      errorBox.textContent = message || '';
      errorBox.style.display = message ? 'block' : 'none';
    }

    function renderPredictions(list) {
      predictionList.innerHTML = '';
      if (!Array.isArray(list) || !list.length) {
        predictionList.innerHTML = '<div class="status-text">No predictions available.</div>';
        return;
      }

      list.forEach((item) => {
        const row = document.createElement('div');
        row.className = 'prediction-item';

        const label = document.createElement('span');
        label.textContent = item.class || 'Unknown';

        const barWrap = document.createElement('div');
        barWrap.className = 'bar';
        const bar = document.createElement('span');
        const percent = Math.max(0, Number(item.confidence || 0) * 100);
        bar.style.width = `${percent}%`;
        barWrap.appendChild(bar);

        const value = document.createElement('strong');
        value.textContent = item.confidence ? `${(Number(item.confidence) * 100).toFixed(1)}%` : '0%';

        row.append(label, barWrap, value);
        predictionList.appendChild(row);
      });
    }

    function updatePreview(file) {
      if (!file) {
        previewImage.style.display = 'none';
        previewPlaceholder.style.display = 'block';
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewImage.style.display = 'block';
        previewPlaceholder.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    imageInput.addEventListener('change', (event) => {
      const file = event.target.files && event.target.files[0];
      selectedFile = file || null;
      fileFeedback.textContent = file ? file.name : 'No file selected';
      updatePreview(file);
      setError('');
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      uploadBox.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadBox.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      uploadBox.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadBox.classList.remove('dragover');
      });
    });

    uploadBox.addEventListener('drop', (event) => {
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) {
        selectedFile = file;
        imageInput.files = event.dataTransfer.files;
        fileFeedback.textContent = file.name;
        updatePreview(file);
      }
    });

    modeButtons.forEach((button) => {
      button.addEventListener('click', () => setMode(button.dataset.mode));
    });

    document.getElementById('resetButton').addEventListener('click', () => {
      selectedFile = null;
      imageInput.value = '';
      fileFeedback.textContent = 'No file selected';
      previewImage.style.display = 'none';
      previewPlaceholder.style.display = 'block';
      predictedClass.textContent = '—';
      confidenceValue.textContent = '—';
      lowConfidence.textContent = '—';
      gridCells.textContent = '—';
      predictionList.innerHTML = '';
      statusText.textContent = 'Upload an image and run a prediction.';
      setError('');
    });

    predictButton.addEventListener('click', async () => {
      if (!selectedFile) {
        setError('Please choose an image before running analysis.');
        return;
      }

      const formData = new FormData();
      formData.append('file', selectedFile);

      if (currentMode === 'heatmap') {
        formData.append('cell_size', document.getElementById('cellSize').value);
        formData.append('overlap', document.getElementById('overlap').value);
        formData.append('tta', String(document.getElementById('ttaToggle').checked));
        formData.append('temperature', document.getElementById('temperature').value);
        formData.append('low_confidence_threshold', document.getElementById('confidenceThreshold').value);
      }

      const endpoint = currentMode === 'heatmap' ? '/analyze' : '/classify';
      predictButton.disabled = true;
      predictButton.textContent = 'Analyzing...';
      statusText.textContent = 'Sending image to the model...';
      setError('');

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
          throw new Error(result.error || 'Analysis failed.');
        }

        predictedClass.textContent = result.predicted_class || '—';
        confidenceValue.textContent = result.confidence || '—';
        lowConfidence.textContent = result.low_confidence === undefined ? '—' : (result.low_confidence ? 'Yes' : 'No');
        gridCells.textContent = result.grid_size || '—';
        renderPredictions(result.top_predictions || []);
        statusText.textContent = currentMode === 'heatmap' ? 'Heatmap analysis completed.' : 'Single-image classification completed.';

        if (result.artifacts && result.artifacts.length) {
          const artifact = result.artifacts.find((item) => item && item.toLowerCase().includes('heatmap')) || result.artifacts[0];
          if (artifact) {
            previewImage.src = artifact;
            previewImage.style.display = 'block';
            previewPlaceholder.style.display = 'none';
          }
        }
      } catch (error) {
        setError(error.message || 'Unexpected error.');
        statusText.textContent = 'Analysis could not complete.';
      } finally {
        predictButton.disabled = false;
        predictButton.textContent = 'Analyze image';
      }
    });
  </script>
</body>
</html>
'''


def _load_image_from_upload(uploaded_file: Image.Image) -> Image.Image:
    return Image.open(uploaded_file.stream).convert('RGB')


def _serialize_top_predictions(top_predictions):
    return [
        {
            'class': item['class'],
            'confidence': f"{item['confidence']:.2%}",
            'index': item['index'],
        }
        for item in top_predictions
    ]


def initialize_model():
    global MODEL, DEVICE, MODEL_PATH
    DEVICE = get_device()
    resolved_path = _resolve_model_path()
    MODEL_PATH = resolved_path

    if not Path(resolved_path).exists():
        print(f'⚠️ Model checkpoint not found at {resolved_path}. Starting in demo mode.')
        MODEL = None
        return False

    MODEL = load_model(MODEL_PATH, device=DEVICE)
    print(f'\u2713 Model loaded successfully on {DEVICE} from {MODEL_PATH}')
    return True


@app.route('/', methods=['GET'])
def check():
    return render_template_string(HTML_TEMPLATE, class_names=CANONICAL_CLASS_NAMES)


@app.route('/classify', methods=['POST'])
def classify():
    try:
        if MODEL is None:
            return jsonify({
                'error': 'Model weights are not available. Add a valid checkpoint such as best_eurosat_model.pth to the project root before running inference.',
                'success': False,
            }), 503

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image = _load_image_from_upload(uploaded_file)
        prediction = predict_single_image(MODEL, image, DEVICE, use_tta=True, temperature=1.0)

        response = {
            'predicted_class': prediction.predicted_class,
            'confidence': f"{prediction.confidence:.2%}",
            'success': True,
            'low_confidence': prediction.low_confidence,
            'top_predictions': _serialize_top_predictions(prediction.top_predictions),
        }
        return jsonify(response)

    except Exception as error:
        return jsonify({'error': str(error), 'success': False}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if MODEL is None:
            return jsonify({
                'error': 'Model weights are not available. Add a valid checkpoint such as best_eurosat_model.pth to the project root before running heatmap analysis.',
                'success': False,
            }), 503

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image = _load_image_from_upload(uploaded_file)

        cell_size = int(request.form.get('cell_size', 128))
        overlap = float(request.form.get('overlap', 0.25))
        use_tta = request.form.get('tta', 'true').lower() in {'1', 'true', 'yes', 'on'}
        temperature = float(request.form.get('temperature', 1.0))
        low_confidence_threshold = float(request.form.get('low_confidence_threshold', 0.55))
        neighbor_k = int(request.form.get('neighbor_k', 8))
        smoothing_alpha = float(request.form.get('smoothing_alpha', 0.65))
        preserve_linear_water = request.form.get('preserve_linear_water', 'true').lower() in {'1', 'true', 'yes', 'on'}
        water_preserve_threshold = float(request.form.get('water_preserve_threshold', 0.35))
        consensus_top_k = int(request.form.get('consensus_top_k', 3))
        consensus_boost = float(request.form.get('consensus_boost', 0.20))
        enable_refinement = request.form.get('enable_refinement', 'true').lower() in {'1', 'true', 'yes', 'on'}

        heatmap_result = predict_heatmap(
            MODEL,
            image,
            DEVICE,
            cell_size=cell_size,
            overlap=overlap,
            use_tta=use_tta,
            temperature=temperature,
            low_confidence_threshold=low_confidence_threshold,
            neighbor_k=neighbor_k,
            smoothing_alpha=smoothing_alpha,
            preserve_linear_water=preserve_linear_water,
            water_preserve_threshold=water_preserve_threshold,
            consensus_top_k=consensus_top_k,
            consensus_boost=consensus_boost,
            enable_refinement=enable_refinement,
        )

        file_stem = Path(uploaded_file.filename).stem or 'upload'
        output_dir = os.path.join('results', 'heatmaps', datetime.now().strftime('%Y%m%d_%H%M%S'))
        render_paths = render_heatmap_outputs(image, heatmap_result, output_dir, prefix=file_stem)

        response = {
            'success': True,
            'predicted_class': heatmap_result['predicted_class'],
            'confidence': f"{heatmap_result['confidence']:.2%}",
            'top_predictions': _serialize_top_predictions(heatmap_result['top_predictions']),
            'grid_size': heatmap_result['grid_size'],
            'cell_size': heatmap_result['cell_size'],
            'overlap': heatmap_result['overlap'],
            'low_confidence_cells': heatmap_result['low_confidence_cells'],
            'low_confidence_ratio': f"{heatmap_result['low_confidence_ratio']:.2%}",
            'artifacts': render_paths,
        }
        return jsonify(response)

    except Exception as error:
        return jsonify({'error': str(error), 'success': False}), 500


if __name__ == '__main__':
    print('=' * 60)
    print('Land Cover Classification API')
    print('=' * 60)
    initialize_model()
    print('\nStarting Flask server on http://localhost:5000')
    print("Use POST /classify for a single prediction or POST /analyze for a heatmap")
    print('=' * 60)
    app.run(debug=False, port=5000)
