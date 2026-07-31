from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template_string, request, send_file

from stages.stage1_single_cell.annotation.annotate_heatmap_cells import has_complete_artifacts, process_image
from stages.stage2_multiclass.taxonomy.class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label
from stages.stage1_single_cell.core.landcover_pipeline import get_device, load_model


def extract_cell_patch(image_path: str, cell_box: List[int]) -> object:
    """Extract a cell patch from an image given its bounding box."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    left, top, right, bottom = cell_box
    return img.crop((left, top, right, bottom))


def save_corrected_cells(annotation_file: Path, reference_image_path: Path, output_dir: Path) -> Dict:
    """Extract corrected cell patches from annotation JSON and save to organized folders.
    
    Returns: {
        'extracted_count': int,
        'failed_count': int,
        'output_dir': str,
        'by_class': {'ClassName': count, ...}
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with annotation_file.open('r') as f:
        rows = json.load(f)
    
    results = {'extracted_count': 0, 'failed_count': 0, 'by_class': {}, 'output_dir': str(output_dir)}
    
    for row in rows:
        if row.get('is_correct', True):
            continue
        
        correct_label = row.get('correct_label', row.get('model_prediction'))
        if not correct_label:
            results['failed_count'] += 1
            continue
        
        class_dir = output_dir / correct_label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            patch = extract_cell_patch(str(reference_image_path), row['box'])
            cell_id = row.get('cell_id', f"{row.get('row', 0)}_{row.get('col', 0)}")
            img_stem = reference_image_path.stem
            output_path = class_dir / f'{img_stem}_cell_{cell_id}.png'
            patch.save(str(output_path))
            
            results['extracted_count'] += 1
            results['by_class'][correct_label] = results['by_class'].get(correct_label, 0) + 1
        except Exception as e:
            results['failed_count'] += 1
    
    return results


def build_app(reference_dir: Path, workspace_root: Path, artifacts_root: Path) -> Flask:
    app = Flask(__name__)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}

    def list_reference_images() -> List[Path]:
        return sorted(
            [p for p in reference_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions],
            key=lambda p: p.name.lower(),
        )

    def annotation_path_for(stem: str) -> Path:
        new_path = artifacts_root / stem / f'{stem}_annotations.json'
        if new_path.exists():
            return new_path
        return workspace_root / f'{stem}_annotations.json'

    def heatmap_path_for(stem: str) -> Path:
        new_path = artifacts_root / stem / f'{stem}_heatmap_grid.png'
        if new_path.exists():
            return new_path
        return workspace_root / f'{stem}_heatmap_grid.png'

    @app.get('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            class_names=CANONICAL_CLASS_NAMES,
        )

    @app.get('/api/images')
    def get_images():
        items: List[Dict] = []
        for image_path in list_reference_images():
            stem = image_path.stem
            annotation_path = annotation_path_for(stem)
            heatmap_path = heatmap_path_for(stem)
            items.append(
                {
                    'stem': stem,
                    'imageName': image_path.name,
                    'annotationExists': annotation_path.exists(),
                    'heatmapExists': heatmap_path.exists(),
                    'imageUrl': f'/api/file/reference/{stem}',
                    'heatmapUrl': f'/api/file/heatmap/{stem}',
                }
            )
        return jsonify({'images': items})

    @app.get('/api/annotation/<path:stem>')
    def load_annotation(stem: str):
        annotation_path = annotation_path_for(stem)
        if not annotation_path.exists():
            return jsonify({'error': f'Annotation file not found: {annotation_path.name}'}), 404

        with annotation_path.open('r', encoding='utf-8') as f:
            rows = json.load(f)

        return jsonify(
            {
                'stem': stem,
                'annotationPath': str(annotation_path),
                'rows': rows,
            }
        )

    @app.post('/api/annotation/<path:stem>')
    def save_annotation(stem: str):
        annotation_path = annotation_path_for(stem)
        if not annotation_path.exists():
            return jsonify({'error': f'Annotation file not found: {annotation_path.name}'}), 404

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or 'rows' not in payload or not isinstance(payload['rows'], list):
            return jsonify({'error': 'Invalid payload. Expected {"rows": [...]}'}), 400

        normalized_rows = []
        for row in payload['rows']:
          if isinstance(row, dict):
            row = dict(row)
            row['model_prediction'] = canonicalize_label(row.get('model_prediction')) or row.get('model_prediction')
            row['correct_label'] = canonicalize_label(row.get('correct_label')) or row.get('correct_label')
            normalized_rows.append(row)

        with annotation_path.open('w', encoding='utf-8') as f:
          json.dump(normalized_rows, f, indent=2)

        return jsonify({'ok': True, 'savedPath': str(annotation_path), 'rowCount': len(normalized_rows)})

    @app.post('/api/extract/<path:stem>')
    def extract_cells(stem: str):
        """Extract corrected cells and save to organized class folders."""
        annotation_path = annotation_path_for(stem)
        if not annotation_path.exists():
            return jsonify({'error': f'Annotation file not found: {annotation_path.name}'}), 404
        
        # Find the reference image
        candidates = [p for p in list_reference_images() if p.stem == stem]
        if not candidates:
            return jsonify({'error': f'Reference image not found for stem: {stem}'}), 404
        
        reference_image_path = candidates[0]
        output_dir = workspace_root / 'data' / 'cell_training' / 'labeled_cells'
        
        try:
            results = save_corrected_cells(annotation_path, reference_image_path, output_dir)
            return jsonify({
                'ok': True,
                'extracted': results['extracted_count'],
                'failed': results['failed_count'],
                'byClass': results['by_class'],
                'outputDir': results['output_dir'],
                'message': f"Extracted {results['extracted_count']} corrected cells for {stem}"
            })
        except Exception as e:
            return jsonify({'error': f'Extraction failed: {str(e)}'}), 500

    @app.get('/api/file/reference/<path:stem>')
    def get_reference_file(stem: str):
        candidates = [p for p in list_reference_images() if p.stem == stem]
        if not candidates:
            return jsonify({'error': f'Reference image not found for stem: {stem}'}), 404
        return send_file(candidates[0])

    @app.get('/api/file/heatmap/<path:stem>')
    def get_heatmap_file(stem: str):
        heatmap_path = heatmap_path_for(stem)
        if not heatmap_path.exists():
            return jsonify({'error': f'Heatmap file not found: {heatmap_path.name}'}), 404
        return send_file(heatmap_path)

    return app


HTML_TEMPLATE = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cell Annotation Web</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #152238;
      --muted: #6a7280;
      --accent: #005e8a;
      --warn: #be123c;
      --line: #d5dbe3;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 0%, #dce9f5 0%, var(--bg) 45%);
    }
    .app {
      display: grid;
      grid-template-columns: 280px 1fr 320px;
      gap: 12px;
      height: 100vh;
      padding: 12px;
      box-sizing: border-box;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      box-sizing: border-box;
      overflow: auto;
    }
    .title {
      margin: 0 0 8px 0;
      font-size: 18px;
      font-weight: 700;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .stack {
      display: grid;
      gap: 8px;
    }
    button, select {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button.warn {
      background: var(--warn);
      color: #fff;
      border-color: var(--warn);
    }
    .list {
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      cursor: pointer;
      background: #fff;
    }
    .item.active {
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
    }
    .viewer {
      display: grid;
      place-items: center;
      background: #eef2f7;
      border-radius: 12px;
      border: 1px solid var(--line);
      min-height: 500px;
      overflow: auto;
    }
    canvas {
      display: block;
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
      background: #fff;
    }
    .kv {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 6px;
      font-size: 14px;
      margin: 8px 0;
    }
    .status {
      padding: 8px;
      border-radius: 8px;
      font-size: 13px;
      background: #f3f4f6;
      border: 1px solid var(--line);
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 60px;
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel">
      <h1 class="title">Images</h1>
      <div class="row">
        <button id="refreshBtn">Refresh</button>
      </div>
      <div id="imageList" class="list"></div>
    </aside>

    <main class="panel">
      <h2 class="title">Cell Grid</h2>
      <div class="row">
        <button id="toggleBaseBtn">Toggle Base Image</button>
        <span id="stats" class="meta"></span>
      </div>
      <div class="viewer">
        <canvas id="canvas"></canvas>
      </div>
    </main>

    <aside class="panel">
      <h2 class="title">Cell Editor</h2>
      <div id="status" class="status">Load an image to begin.</div>

      <div class="kv">
        <div>Cell</div><div id="cellId">-</div>
        <div>Final</div><div id="predicted">-</div>
        <div>Raw</div><div id="rawPredicted">-</div>
        <div>Saved</div><div id="savedLabel">-</div>
        <div>Raw Conf.</div><div id="rawConfidence">-</div>
        <div>Confidence</div><div id="confidence">-</div>
      </div>

      <div class="stack">
        <label>Correct Label</label>
        <select id="labelSelect">
          {% for name in class_names %}
          <option value="{{ name }}">{{ name }}</option>
          {% endfor %}
        </select>

        <label class="row" style="margin: 0;">
          <input id="isCorrectCheckbox" type="checkbox" checked />
          <span>Cell is correct</span>
        </label>

        <button id="applyBtn" class="primary">Apply To Selected Cell</button>
        <button id="saveBtn" class="primary">Save & Extract Cells</button>
        <button id="markIncorrectBtn" class="warn">Mark Selected Incorrect</button>
      </div>
    </aside>
  </div>

<script>
let images = [];
let activeStem = null;
let rows = [];
let selectedIndex = -1;
let showHeatmap = true;

const imageList = document.getElementById('imageList');
const statusEl = document.getElementById('status');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statsEl = document.getElementById('stats');
const cellIdEl = document.getElementById('cellId');
const predictedEl = document.getElementById('predicted');
const rawPredictedEl = document.getElementById('rawPredicted');
const confidenceEl = document.getElementById('confidence');
const rawConfidenceEl = document.getElementById('rawConfidence');
const labelSelect = document.getElementById('labelSelect');
const isCorrectCheckbox = document.getElementById('isCorrectCheckbox');

const baseImage = new Image();
const heatmapImage = new Image();

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Request failed');
  }
  return data;
}

function renderImageList() {
  imageList.innerHTML = '';
  images.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'item' + (item.stem === activeStem ? ' active' : '');
    div.innerHTML = `
      <div><strong>${item.imageName}</strong></div>
      <div class="meta">annotations: ${item.annotationExists ? 'yes' : 'no'} | heatmap: ${item.heatmapExists ? 'yes' : 'no'}</div>
    `;
    div.onclick = () => loadStem(item.stem);
    imageList.appendChild(div);
  });
}

function draw() {
  if (!baseImage.complete || baseImage.naturalWidth === 0) {
    return;
  }

  canvas.width = baseImage.naturalWidth;
  canvas.height = baseImage.naturalHeight;

  ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);

  rows.forEach((row, idx) => {
    const [left, top, right, bottom] = row.box;
    const isSelected = idx === selectedIndex;
    const isCorrect = row.is_correct !== false;
    const correctionLabel = row.correct_label && row.correct_label !== row.model_prediction ? row.correct_label : '';
    const modelLabel = `${row.model_prediction} (${(row.model_confidence * 100).toFixed(0)}%)`;

    ctx.lineWidth = isSelected ? 3 : 1.5;
    ctx.strokeStyle = isSelected ? '#00c2ff' : (isCorrect ? 'rgba(20,20,20,0.35)' : 'rgba(220,30,80,0.9)');
    ctx.strokeRect(left, top, right - left, bottom - top);

    ctx.font = '11px Segoe UI';
    const modelWidth = ctx.measureText(modelLabel).width;
    ctx.fillStyle = 'rgba(0,0,0,0.62)';
    ctx.fillRect(left + 2, top + 2, modelWidth + 8, 16);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(modelLabel, left + 6, top + 14);

    if (!isCorrect) {
      ctx.fillStyle = 'rgba(220,30,80,0.16)';
      ctx.fillRect(left, top, right - left, bottom - top);

      if (correctionLabel) {
        const label = `â†’ ${correctionLabel}`;
        const textWidth = ctx.measureText(label).width;
        const labelX = left + 2;
        const labelY = bottom - 4;
        ctx.fillStyle = 'rgba(220,30,80,0.92)';
        ctx.fillRect(labelX - 1, labelY - 13, textWidth + 6, 15);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, labelX + 2, labelY - 1);
      }
    }
  });

  const incorrectCount = rows.filter(r => r.is_correct === false).length;
  statsEl.textContent = `${rows.length} cells | ${incorrectCount} marked incorrect`;
}

function findCellAt(x, y) {
  for (let i = 0; i < rows.length; i++) {
    const [l, t, r, b] = rows[i].box;
    if (x >= l && x <= r && y >= t && y <= b) {
      return i;
    }
  }
  return -1;
}

function updateEditor() {
  if (selectedIndex < 0 || selectedIndex >= rows.length) {
    cellIdEl.textContent = '-';
    predictedEl.textContent = '-';
    rawPredictedEl.textContent = '-';
    document.getElementById('savedLabel').textContent = '-';
    confidenceEl.textContent = '-';
    rawConfidenceEl.textContent = '-';
    return;
  }
  const row = rows[selectedIndex];
  cellIdEl.textContent = row.cell_id;
  predictedEl.textContent = row.model_prediction;
  rawPredictedEl.textContent = row.raw_prediction || row.model_prediction;
  document.getElementById('savedLabel').textContent = row.is_correct === false ? row.correct_label : 'Same as model';
  confidenceEl.textContent = `${(row.model_confidence * 100).toFixed(1)}%`;
  rawConfidenceEl.textContent = row.raw_confidence != null ? `${(row.raw_confidence * 100).toFixed(1)}%` : 'n/a';
  labelSelect.value = row.correct_label;
  isCorrectCheckbox.checked = row.is_correct !== false;
}

async function loadImages() {
  const data = await fetchJson('/api/images');
  images = data.images;
  if (!activeStem && images.length) {
    activeStem = images[0].stem;
  }
  renderImageList();
  if (activeStem) {
    await loadStem(activeStem);
  }
}

async function loadStem(stem) {
  activeStem = stem;
  renderImageList();
  setStatus(`Loading ${stem}...`);

  const info = images.find(i => i.stem === stem);
  if (!info) {
    setStatus(`Image not found: ${stem}`);
    return;
  }

  if (!info.annotationExists) {
    rows = [];
    selectedIndex = -1;
    updateEditor();
    draw();
    setStatus(`Missing annotation JSON for ${stem}. Run annotate_heatmap_cells first.`);
    return;
  }

  const annotation = await fetchJson(`/api/annotation/${encodeURIComponent(stem)}`);
  rows = annotation.rows;
  selectedIndex = -1;

  const imageLoaded = new Promise((resolve, reject) => {
    baseImage.onload = resolve;
    baseImage.onerror = reject;
  });
  baseImage.src = `/api/file/reference/${encodeURIComponent(stem)}?t=${Date.now()}`;

  await imageLoaded;

  updateEditor();
  draw();
  setStatus(`Loaded ${stem}. Click a cell to edit.`);
}

canvas.addEventListener('click', (event) => {
  if (!rows.length) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  selectedIndex = findCellAt(x, y);
  updateEditor();
  draw();
});

document.getElementById('refreshBtn').onclick = async () => {
  try {
    await loadImages();
    setStatus('Refreshed image list.');
  } catch (err) {
    setStatus(err.message);
  }
};

document.getElementById('toggleBaseBtn').onclick = () => {
  showHeatmap = !showHeatmap;
  draw();
};

document.getElementById('applyBtn').onclick = () => {
  if (selectedIndex < 0) {
    setStatus('Select a cell first.');
    return;
  }
  const row = rows[selectedIndex];
  row.correct_label = labelSelect.value;
  row.is_correct = isCorrectCheckbox.checked;
  updateEditor();
  draw();
  setStatus(`Updated cell ${row.cell_id}.`);
};

document.getElementById('markIncorrectBtn').onclick = () => {
  if (selectedIndex < 0) {
    setStatus('Select a cell first.');
    return;
  }
  const row = rows[selectedIndex];
  row.is_correct = false;
  isCorrectCheckbox.checked = false;
  updateEditor();
  draw();
  setStatus(`Marked cell ${row.cell_id} as incorrect.`);
};

document.getElementById('saveBtn').onclick = async () => {
  if (!activeStem) {
    setStatus('No active image loaded.');
    return;
  }
  try {
    setStatus('Saving and extracting cells...');
    const result = await fetchJson(`/api/annotation/${encodeURIComponent(activeStem)}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({rows}),
    });
    
    // Now extract the corrected cells
    const extractResult = await fetchJson(`/api/extract/${encodeURIComponent(activeStem)}`, {
      method: 'POST',
    });
    
    let msg = `âœ… Saved ${result.rowCount} annotations for ${activeStem}.\n`;
    msg += `âœ… Extracted ${extractResult.extracted} corrected cells.\n`;
    if (extractResult.byClass && Object.keys(extractResult.byClass).length > 0) {
      msg += `Classes: ${Object.entries(extractResult.byClass).map(([c, cnt]) => `${c}(${cnt})`).join(', ')}`;
    }
    msg += `\nðŸ“ Saved to: ${extractResult.outputDir}`;
    setStatus(msg);
  } catch (err) {
    setStatus(`Failed: ${err.message}`);
  }
};

loadImages().catch((err) => setStatus(err.message));
</script>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description='Web UI for cell annotation JSON edits.')
    parser.add_argument('--reference-dir', default='data/cell_training/reference_images', help='Folder containing reference images.')
    parser.add_argument('--workspace-root', default='.', help='Workspace root for project-relative paths.')
    parser.add_argument('--artifacts-root', default='results/heatmaps/image', help='Per-image artifacts root created by annotate_heatmap_cells.')
    parser.add_argument('--model', default='best_eurosat_model.pth', help='Model used for auto-generating missing artifacts.')
    parser.add_argument('--cell-size', type=int, default=128, help='Cell size used when auto-generating missing artifacts.')
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap used when auto-generating missing artifacts.')
    parser.add_argument('--disable-refinement', action='store_true', help='Use raw predictions without neighbor refinement when auto-generating.')
    parser.add_argument('--no-auto-prepare-missing', action='store_true', help='Disable incremental generation of missing artifacts at startup.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the web server to.')
    parser.add_argument('--port', type=int, default=8765, help='Port for the web server.')
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    artifacts_root = (workspace_root / args.artifacts_root).resolve() if not Path(args.artifacts_root).is_absolute() else Path(args.artifacts_root).resolve()

    if not reference_dir.exists():
        raise FileNotFoundError(f'Reference directory not found: {reference_dir}')

    artifacts_root.mkdir(parents=True, exist_ok=True)

    if not args.no_auto_prepare_missing:
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}
        image_paths = sorted(
            [p for p in reference_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions],
            key=lambda p: p.name.lower(),
        )
        missing_images = [p for p in image_paths if not has_complete_artifacts({
            'heatmap_grid': artifacts_root / p.stem / f'{p.stem}_heatmap_grid.png',
            'annotations': artifacts_root / p.stem / f'{p.stem}_annotations.json',
            'overlay': artifacts_root / p.stem / f'{p.stem}_overlay.png',
            'confidence': artifacts_root / p.stem / f'{p.stem}_confidence.png',
            'uncertainty': artifacts_root / p.stem / f'{p.stem}_uncertainty.png',
            'report': artifacts_root / p.stem / f'{p.stem}_report.png',
            'heatmap_json': artifacts_root / p.stem / f'{p.stem}_heatmap.json',
            'image_dir': artifacts_root / p.stem,
        })]

        if missing_images:
            print(f'Preparing missing artifacts for {len(missing_images)} image(s)...')
            device = get_device()
            model = load_model(args.model, device=device)
            for image_path in missing_images:
                print(f'  Generating: {image_path.name}')
                process_image(
                    image_path=image_path,
                    model=model,
                    device=device,
                    artifacts_root=artifacts_root,
                    output_dir=str(workspace_root / 'data' / 'cell_training' / 'labeled_cells'),
                    cell_size=args.cell_size,
                    overlap=args.overlap,
                    disable_refinement=args.disable_refinement,
                    skip_extraction=True,
                    skip_existing=True,
                )
        else:
            print('All reference images already have complete artifacts.')

    app = build_app(reference_dir=reference_dir, workspace_root=workspace_root, artifacts_root=artifacts_root)
    print('Cell annotation web UI running.')
    print(f'Open http://{args.host}:{args.port} in your browser.')
    print('Click a cell, choose label from dropdown, then click Save JSON.')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

