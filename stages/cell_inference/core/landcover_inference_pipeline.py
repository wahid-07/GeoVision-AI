"""
Shared land-cover inference utilities.

This module centralizes model loading, correct probability conversion for the
LogSoftmax head, single-image classification, and patch-grid heatmap generation.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from torchvision import models, transforms

from stages.landcover_model.taxonomy.landcover_class_taxonomy import (
    CANONICAL_CLASS_COLORS,
    CANONICAL_CLASS_NAMES,
    RAW_CLASS_LABELS,
    RAW_TO_CANONICAL_INDEX,
)

RESAMPLE_BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, 'Resampling', Image).NEAREST


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _build_backbone_model(num_classes: int = 10) -> nn.Module:
    model = models.wide_resnet50_2(pretrained=False)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1),
    )
    return model


class LandCoverWrapper(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.network = _build_backbone_model(num_classes)

    def forward(self, xb):
        return self.network(xb)


def load_model(model_path: str, device: Optional[torch.device] = None, num_classes: int = 10) -> nn.Module:
    """Load checkpoint and infer the architecture from state_dict keys."""
    if device is None:
        device = get_device()

    state_dict = torch.load(model_path, map_location=device)

    inferred_num_classes = num_classes
    class_bias_sizes = [
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if 'fc' in key and key.endswith('bias') and hasattr(tensor, 'numel')
    ]
    if class_bias_sizes:
        inferred_num_classes = min(class_bias_sizes)
    
    # Detect architecture from state_dict keys
    has_network_wrapper = any(k.startswith('network.') for k in state_dict.keys())
    has_sequential_fc = any(k.startswith('fc.0.') or k.startswith('fc.1.') or k.startswith('fc.2.') or k.startswith('fc.3.') for k in state_dict.keys())
    has_simple_fc = 'fc.weight' in state_dict and 'fc.bias' in state_dict
    
    candidates: List[nn.Module] = []
    
    # Order candidates by likelihood
    if has_network_wrapper:
        candidates.append(LandCoverWrapper(num_classes=inferred_num_classes))
    if has_sequential_fc or not has_simple_fc:
        candidates.append(_build_backbone_model(num_classes=inferred_num_classes))
    if has_simple_fc:
        model = models.wide_resnet50_2(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, inferred_num_classes)
        candidates.append(model)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            candidate.load_state_dict(state_dict)
            candidate.to(device)
            candidate.eval()
            return candidate
        except Exception as error:
            last_error = error

    raise RuntimeError(f'Unable to load model checkpoint from {model_path}: {last_error}')


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_image(image_source) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert('RGB')
    return Image.open(image_source).convert('RGB')


def preprocess_patch(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    transform = build_transform(image_size=image_size)
    return transform(image).unsqueeze(0)


def _apply_tta_variants(image: Image.Image) -> List[Image.Image]:
    return [
        image,
        ImageOps.mirror(image),
        ImageOps.flip(image),
        ImageOps.mirror(ImageOps.flip(image)),
    ]


def log_probs_to_probs(log_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError('temperature must be greater than zero')

    scaled = log_probs / temperature
    probs = torch.exp(scaled)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs


def canonicalize_probabilities(raw_probs: torch.Tensor) -> torch.Tensor:
    if raw_probs.dim() == 1:
        raw_probs = raw_probs.unsqueeze(0)

    if raw_probs.size(1) == len(CANONICAL_CLASS_NAMES):
        return raw_probs / raw_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    if raw_probs.size(1) < max(RAW_TO_CANONICAL_INDEX.keys()) + 1:
        raise ValueError(
            f'Expected either {len(CANONICAL_CLASS_NAMES)} merged classes or {max(RAW_TO_CANONICAL_INDEX.keys()) + 1} raw classes, '
            f'got {raw_probs.size(1)}.'
        )

    canonical = torch.zeros((raw_probs.size(0), len(CANONICAL_CLASS_NAMES)), dtype=raw_probs.dtype, device=raw_probs.device)
    for raw_idx, canonical_idx in RAW_TO_CANONICAL_INDEX.items():
        canonical[:, canonical_idx] += raw_probs[:, raw_idx]
    canonical = canonical / canonical.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return canonical


def get_top_predictions(probabilities: torch.Tensor, top_n: int = 3) -> List[Dict[str, object]]:
    values, indices = torch.topk(probabilities, k=min(top_n, probabilities.numel()))
    return [
        {
            'class': CANONICAL_CLASS_NAMES[index.item()],
            'confidence': float(value.item()),
            'index': int(index.item()),
        }
        for value, index in zip(values, indices)
    ]


def _top_predictions_for_cell(probabilities: torch.Tensor, top_n: int = 3) -> List[Dict[str, object]]:
    return get_top_predictions(probabilities, top_n=top_n)


@dataclass
class SinglePrediction:
    predicted_class: str
    confidence: float
    probabilities: torch.Tensor
    top_predictions: List[Dict[str, object]]
    low_confidence: bool


def predict_single_image(
    model: nn.Module,
    image_source,
    device: torch.device,
    image_size: int = 224,
    use_tta: bool = True,
    temperature: float = 1.0,
    low_confidence_threshold: float = 0.55,
) -> SinglePrediction:
    image = load_image(image_source)
    variants = _apply_tta_variants(image) if use_tta else [image]

    variant_probs: List[torch.Tensor] = []
    for variant in variants:
        image_tensor = preprocess_patch(variant, image_size=image_size).to(device)
        with torch.no_grad():
            log_probs = model(image_tensor)
            probs = log_probs_to_probs(log_probs, temperature=temperature)
            canonical_probs = canonicalize_probabilities(probs)[0].detach().cpu()
            variant_probs.append(canonical_probs)

    merged_probs = torch.stack(variant_probs, dim=0).mean(dim=0)
    confidence, predicted_idx = torch.max(merged_probs, dim=0)
    return SinglePrediction(
        predicted_class=CANONICAL_CLASS_NAMES[predicted_idx.item()],
        confidence=float(confidence.item()),
        probabilities=merged_probs,
        top_predictions=get_top_predictions(merged_probs, top_n=3),
        low_confidence=float(confidence.item()) < low_confidence_threshold,
    )


def _grid_positions(length: int, cell_size: int, overlap: float) -> List[int]:
    if cell_size <= 0:
        raise ValueError('cell_size must be greater than zero')

    overlap = min(max(overlap, 0.0), 0.9)
    step = max(1, int(round(cell_size * (1.0 - overlap))))
    last_start = max(length - cell_size, 0)
    positions = list(range(0, last_start + 1, step))
    if not positions:
        positions = [0]
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def _image_entropy(probabilities: torch.Tensor) -> float:
    return float((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).item())


def _refine_with_neighbors(
    probs: torch.Tensor,
    centers: Sequence[Tuple[float, float]],
    image_size: Tuple[int, int],
    low_confidence_threshold: float,
    neighbor_k: int,
    smoothing_alpha: float,
    preserve_classes: Optional[Sequence[int]] = None,
    preserve_threshold: float = 0.35,
    consensus_top_k: int = 3,
    consensus_boost: float = 0.20,
) -> Tuple[torch.Tensor, List[bool]]:
    if probs.size(0) != len(centers):
        raise ValueError('probabilities and centers must have the same length')

    width, height = image_size
    normalized_centers = np.array(
        [(x / max(width, 1), y / max(height, 1)) for x, y in centers],
        dtype=np.float32,
    )
    refined = probs.clone()
    changed_flags: List[bool] = [False] * probs.size(0)
    preserve_class_set = set(preserve_classes or [])

    # Two-pass refinement improves robustness on uncertain cells while avoiding
    # excessive smoothing of confident regions.
    for _ in range(2):
        source_probs = refined.clone()
        confidences = source_probs.max(dim=1).values.cpu().numpy()

        for index in range(probs.size(0)):
            source_top = int(source_probs[index].argmax().item())
            if source_top in preserve_class_set and confidences[index] >= preserve_threshold:
                continue

            if confidences[index] >= low_confidence_threshold:
                continue

            spatial_dist = np.linalg.norm(normalized_centers - normalized_centers[index], axis=1)
            prob_dist = torch.norm(source_probs - source_probs[index], dim=1).cpu().numpy()
            combined_dist = spatial_dist + 0.7 * prob_dist

            neighbor_order = np.argsort(combined_dist)
            neighbor_order = [candidate for candidate in neighbor_order if candidate != index][: max(1, neighbor_k)]
            if not neighbor_order:
                continue

            spatial_tensor = torch.tensor(spatial_dist[neighbor_order], dtype=probs.dtype, device=probs.device)
            prob_tensor = torch.tensor(prob_dist[neighbor_order], dtype=probs.dtype, device=probs.device)
            neighbor_probs = source_probs[neighbor_order]
            neighbor_conf = neighbor_probs.max(dim=1).values

            neighbor_weights = torch.exp(-spatial_tensor * 6.0) * torch.exp(-prob_tensor * 4.0)
            neighbor_weights = neighbor_weights * neighbor_conf.clamp_min(1e-6)
            weight_sum = neighbor_weights.sum().clamp_min(1e-12)
            neighbor_average = (neighbor_probs * neighbor_weights.unsqueeze(1)).sum(dim=0) / weight_sum

            blended = (1.0 - smoothing_alpha) * source_probs[index] + smoothing_alpha * neighbor_average

            # Explicitly include top-k class consensus so strong 2nd/3rd
            # candidates are retained rather than being washed out by dominant neighbors.
            top_k = max(1, min(int(consensus_top_k), source_probs[index].numel()))
            source_topk_idx = torch.topk(source_probs[index], k=top_k).indices
            support = torch.zeros_like(source_probs[index])
            for n_idx in range(neighbor_probs.size(0)):
                neighbor_topk_idx = torch.topk(neighbor_probs[n_idx], k=top_k).indices
                support[neighbor_topk_idx] += neighbor_weights[n_idx]
            support_sum = support.sum().clamp_min(1e-12)
            support = support / support_sum

            candidate_mask = torch.zeros_like(source_probs[index])
            candidate_mask[source_topk_idx] = 1.0
            consensus_prior = support * candidate_mask
            consensus_prior_sum = consensus_prior.sum().clamp_min(1e-12)
            consensus_prior = consensus_prior / consensus_prior_sum

            boost = min(max(float(consensus_boost), 0.0), 0.95)
            blended = (1.0 - boost) * blended + boost * consensus_prior
            refined[index] = blended / blended.sum().clamp_min(1e-12)
            changed_flags[index] = True

    return refined, changed_flags


def predict_heatmap(
    model: nn.Module,
    image_source,
    device: torch.device,
    cell_size: int = 128,
    overlap: float = 0.25,
    use_tta: bool = True,
    temperature: float = 1.0,
    low_confidence_threshold: float = 0.55,
    neighbor_k: int = 8,
    smoothing_alpha: float = 0.65,
    preserve_linear_water: bool = True,
    water_preserve_threshold: float = 0.35,
    consensus_top_k: int = 3,
    consensus_boost: float = 0.20,
    enable_refinement: bool = True,
    image_size: int = 224,
) -> Dict[str, object]:
    image = load_image(image_source)
    width, height = image.size
    x_positions = _grid_positions(width, cell_size, overlap)
    y_positions = _grid_positions(height, cell_size, overlap)

    cells: List[Dict[str, object]] = []
    raw_probs: List[torch.Tensor] = []
    centers: List[Tuple[float, float]] = []
    
    total_cells = len(y_positions) * len(x_positions)
    processed_cells = 0

    progress_bar = tqdm(total=total_cells, desc='Generating heatmap', unit='cell')
    try:
        for row_index, top in enumerate(y_positions):
            for col_index, left in enumerate(x_positions):
                right = min(left + cell_size, width)
                bottom = min(top + cell_size, height)
                patch = image.crop((left, top, right, bottom))

                variants = _apply_tta_variants(patch) if use_tta else [patch]
                patch_variant_probs: List[torch.Tensor] = []
                for variant in variants:
                    patch_tensor = preprocess_patch(variant, image_size=image_size).to(device)
                    with torch.no_grad():
                        log_probs = model(patch_tensor)
                        probs = log_probs_to_probs(log_probs, temperature=temperature)
                        patch_variant_probs.append(canonicalize_probabilities(probs)[0].detach().cpu())

                patch_probs = torch.stack(patch_variant_probs, dim=0).mean(dim=0)
                raw_probs.append(patch_probs)
                centers.append(((left + right) / 2.0, (top + bottom) / 2.0))
                cells.append(
                    {
                        'row': row_index,
                        'col': col_index,
                        'box': [int(left), int(top), int(right), int(bottom)],
                        'center': [float((left + right) / 2.0), float((top + bottom) / 2.0)],
                        'raw_probabilities': patch_probs.tolist(),
                    }
                )
                processed_cells += 1
                progress_bar.update(1)
    finally:
        progress_bar.close()

    preserve_classes: List[int] = []
    if preserve_linear_water and 'WaterBodies' in CANONICAL_CLASS_NAMES:
        preserve_classes.append(CANONICAL_CLASS_NAMES.index('WaterBodies'))

    raw_probs_tensor = torch.stack(raw_probs, dim=0)
    if enable_refinement:
        refined_probs_tensor, changed_flags = _refine_with_neighbors(
            raw_probs_tensor,
            centers,
            image_size=(width, height),
            low_confidence_threshold=low_confidence_threshold,
            neighbor_k=neighbor_k,
            smoothing_alpha=smoothing_alpha,
            preserve_classes=preserve_classes,
            preserve_threshold=water_preserve_threshold,
            consensus_top_k=consensus_top_k,
            consensus_boost=consensus_boost,
        )
    else:
        refined_probs_tensor = raw_probs_tensor.clone()
        changed_flags = [False] * raw_probs_tensor.size(0)

    grid = np.zeros((len(y_positions), len(x_positions)), dtype=np.int64)
    confidence_grid = np.zeros_like(grid, dtype=np.float32)
    entropy_grid = np.zeros_like(grid, dtype=np.float32)
    low_confidence_count = 0
    switch_counter: Counter = Counter()

    for index, cell in enumerate(cells):
        raw_probabilities = raw_probs_tensor[index]
        raw_confidence, raw_predicted_idx = torch.max(raw_probabilities, dim=0)
        probabilities = refined_probs_tensor[index]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        entropy = _image_entropy(probabilities)
        margin_values = torch.topk(probabilities, k=min(2, probabilities.numel())).values
        margin = float((margin_values[0] - margin_values[1]).item()) if margin_values.numel() > 1 else float(margin_values[0].item())
        low_confidence = float(confidence.item()) < low_confidence_threshold
        switched_by_refinement = int(raw_predicted_idx.item()) != int(predicted_idx.item())

        if switched_by_refinement:
            source_class = CANONICAL_CLASS_NAMES[raw_predicted_idx.item()]
            target_class = CANONICAL_CLASS_NAMES[predicted_idx.item()]
            switch_counter[(source_class, target_class)] += 1

        if low_confidence:
            low_confidence_count += 1

        cell.update(
            {
                'predicted_index': int(predicted_idx.item()),
                'predicted_class': CANONICAL_CLASS_NAMES[predicted_idx.item()],
                'confidence': float(confidence.item()),
                'raw_predicted_index': int(raw_predicted_idx.item()),
                'raw_predicted_class': CANONICAL_CLASS_NAMES[raw_predicted_idx.item()],
                'raw_confidence': float(raw_confidence.item()),
                'raw_top_predictions': _top_predictions_for_cell(raw_probabilities, top_n=3),
                'final_top_predictions': _top_predictions_for_cell(probabilities, top_n=3),
                'switched_by_refinement': switched_by_refinement,
                'entropy': float(entropy),
                'margin': margin,
                'refined_with_neighbors': bool(changed_flags[index]),
                'low_confidence': low_confidence,
                'final_probabilities': probabilities.tolist(),
            }
        )
        row = cell['row']
        col = cell['col']
        grid[row, col] = int(predicted_idx.item())
        confidence_grid[row, col] = float(confidence.item())
        entropy_grid[row, col] = float(entropy)

    final_probs = refined_probs_tensor.mean(dim=0)
    predicted_class_index = int(final_probs.argmax().item())
    top_predictions = get_top_predictions(final_probs, top_n=3)
    switch_pairs = [
        {
            'from': source,
            'to': target,
            'count': int(count),
        }
        for (source, target), count in switch_counter.most_common(8)
    ]

    return {
        'image_size': [width, height],
        'grid_size': [len(y_positions), len(x_positions)],
        'cell_size': cell_size,
        'overlap': overlap,
        'cells': cells,
        'grid': grid.tolist(),
        'confidence_grid': confidence_grid.tolist(),
        'entropy_grid': entropy_grid.tolist(),
        'final_probabilities': final_probs.tolist(),
        'predicted_class': CANONICAL_CLASS_NAMES[predicted_class_index],
        'predicted_index': predicted_class_index,
        'confidence': float(final_probs.max().item()),
        'top_predictions': top_predictions,
        'low_confidence_cells': low_confidence_count,
        'low_confidence_ratio': float(low_confidence_count / max(len(cells), 1)),
        'refinement_summary': {
            'cells_total': int(len(cells)),
            'cells_refined': int(sum(1 for flag in changed_flags if flag)),
            'cells_switched_class': int(sum(item['count'] for item in switch_pairs)),
            'top_switch_pairs': switch_pairs,
        },
        'refinement_method': 'knn_spatial_topk_consensus_with_water_guard' if enable_refinement else 'disabled',
        'refinement_enabled': bool(enable_refinement),
        'consensus': {
            'top_k': int(consensus_top_k),
            'boost': float(consensus_boost),
        },
        'water_guard': {
            'enabled': preserve_linear_water,
            'classes': [CANONICAL_CLASS_NAMES[index] for index in preserve_classes],
            'preserve_threshold': water_preserve_threshold,
        },
    }
def render_heatmap_outputs(
    image_source,
    heatmap_result: Dict[str, object],
    output_dir: str,
    prefix: Optional[str] = None,
) -> Dict[str, str]:
    image = load_image(image_source)
    os.makedirs(output_dir, exist_ok=True)

    if prefix is None:
        prefix = datetime.now().strftime('%Y%m%d_%H%M%S')

    class_palette = np.array([tuple(int(CANONICAL_CLASS_COLORS[name][i:i + 2], 16) for i in (1, 3, 5)) for name in CANONICAL_CLASS_NAMES], dtype=np.uint8)
    confidence_palette = plt.get_cmap('viridis')

    grid = np.array(heatmap_result['grid'], dtype=np.int64)
    confidence_grid = np.array(heatmap_result['confidence_grid'], dtype=np.float32)
    entropy_grid = np.array(heatmap_result['entropy_grid'], dtype=np.float32)

    class_image = class_palette[grid]
    class_image = Image.fromarray(class_image.astype(np.uint8), mode='RGB').resize(image.size, RESAMPLE_NEAREST)
    overlay = Image.blend(image.convert('RGBA'), class_image.convert('RGBA'), alpha=0.45)

    confidence_image = (confidence_palette(np.clip(confidence_grid, 0.0, 1.0))[:, :, :3] * 255.0).astype(np.uint8)
    confidence_image = Image.fromarray(confidence_image, mode='RGB').resize(image.size, RESAMPLE_NEAREST)

    max_entropy = math.log(len(CANONICAL_CLASS_NAMES))
    entropy_normalized = np.clip(entropy_grid / max(max_entropy, 1e-12), 0.0, 1.0)
    entropy_image = (plt.get_cmap('magma')(entropy_normalized)[:, :, :3] * 255.0).astype(np.uint8)
    entropy_image = Image.fromarray(entropy_image, mode='RGB').resize(image.size, RESAMPLE_NEAREST)

    overlay_path = os.path.join(output_dir, f'{prefix}_overlay.png')
    confidence_path = os.path.join(output_dir, f'{prefix}_confidence.png')
    entropy_path = os.path.join(output_dir, f'{prefix}_uncertainty.png')
    report_path = os.path.join(output_dir, f'{prefix}_report.png')
    json_path = os.path.join(output_dir, f'{prefix}_heatmap.json')

    legend_handles = [Patch(facecolor=CANONICAL_CLASS_COLORS[name], edgecolor='black', label=name) for name in CANONICAL_CLASS_NAMES]

    overlay_figure, overlay_axis = plt.subplots(figsize=(11, 8))
    overlay_axis.imshow(overlay)
    overlay_axis.set_title('Overlay + Class Legend')
    overlay_axis.axis('off')
    overlay_axis.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title='Class Colors',
    )
    overlay_figure.tight_layout()
    overlay_figure.savefig(overlay_path, dpi=220, bbox_inches='tight')
    plt.close(overlay_figure)

    confidence_image.save(confidence_path)
    entropy_image.save(entropy_path)

    uncertainty_decode = {
        'metric': 'normalized_entropy',
        'range': [0.0, 1.0],
        'colormap': 'magma (dark=more certain, bright=more uncertain)',
        'bands': [
            {'min': 0.0, 'max': 0.2, 'label': 'Very certain'},
            {'min': 0.2, 'max': 0.4, 'label': 'Confident'},
            {'min': 0.4, 'max': 0.6, 'label': 'Mixed confidence'},
            {'min': 0.6, 'max': 0.8, 'label': 'Uncertain'},
            {'min': 0.8, 'max': 1.0, 'label': 'Highly uncertain'},
        ],
    }
    json_payload = dict(heatmap_result)
    json_payload['uncertainty_decode'] = uncertainty_decode

    with open(json_path, 'w', encoding='utf-8') as file_handle:
        json.dump(json_payload, file_handle, indent=2)

    figure, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('Overlay + Class Legend')
    axes[0, 1].axis('off')
    axes[0, 1].legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title='Class Colors',
    )

    confidence_axis_image = axes[1, 0].imshow(confidence_grid, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[1, 0].set_title('Confidence Map (Higher is Better)')
    axes[1, 0].axis('off')
    figure.colorbar(confidence_axis_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    entropy_axis_image = axes[1, 1].imshow(entropy_normalized, cmap='magma', vmin=0.0, vmax=1.0)
    axes[1, 1].set_title('Uncertainty Map (Entropy)')
    axes[1, 1].axis('off')
    figure.colorbar(entropy_axis_image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].text(
        0.0,
        -0.18,
        'Decode: 0.0-0.2 Very certain | 0.2-0.4 Confident | 0.4-0.6 Mixed | 0.6-0.8 Uncertain | 0.8-1.0 Highly uncertain',
        transform=axes[1, 1].transAxes,
        fontsize=9,
        ha='left',
        va='top',
        bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '#888888'},
    )

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 1.0])
    figure.savefig(report_path, dpi=220, bbox_inches='tight')
    plt.close(figure)

    return {
        'overlay_path': overlay_path,
        'confidence_path': confidence_path,
        'uncertainty_path': entropy_path,
        'report_path': report_path,
        'json_path': json_path,
    }

