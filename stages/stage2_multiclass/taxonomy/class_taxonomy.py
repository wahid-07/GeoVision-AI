"""Shared class taxonomy for merged land-cover labels.

This module centralizes the merged taxonomy so the rest of the repo can use
one consistent set of class names:

- Crop: AnnualCrop + PermanentCrop
- Urban: Residential + Industrial
- WaterBodies: River + SeaLake

The raw model still emits 10 logits, so we also keep the raw label/index
mapping for inference-time probability merging.
"""

from __future__ import annotations

from typing import Dict, List


RAW_CLASS_LABELS: Dict[int, str] = {
    0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'River',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake',
}

CANONICAL_CLASS_NAMES: List[str] = [
    'Crop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Urban',
    'WaterBodies',
]

RAW_TO_CANONICAL_INDEX: Dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 0,
    7: 4,
    8: 5,
    9: 5,
}

CANONICAL_CLASS_COLORS: Dict[str, str] = {
    'Crop': '#d73027',
    'Forest': '#1a9850',
    'HerbaceousVegetation': '#fee08b',
    'Highway': '#2166ac',
    'Urban': '#f46d43',
    'WaterBodies': '#00acc1',
}

CLASS_ALIASES: Dict[str, List[str]] = {
    'Crop': ['AnnualCrop', 'PermanentCrop'],
    'Forest': ['Forest'],
    'HerbaceousVegetation': ['HerbaceousVegetation'],
    'Highway': ['Highway'],
    'Urban': ['Industrial', 'Residential'],
    'WaterBodies': ['River', 'SeaLake'],
}


def canonicalize_label(label: str | None) -> str | None:
    """Map any legacy label or alias to the merged canonical label."""
    if not label:
        return None

    normalized = label.replace(' ', '').replace('-', '').strip().lower()
    for canonical_name, aliases in CLASS_ALIASES.items():
        if normalized == canonical_name.replace(' ', '').replace('-', '').lower():
            return canonical_name
        for alias in aliases:
            if normalized == alias.replace(' ', '').replace('-', '').lower():
                return canonical_name
    return None


def canonical_folders() -> List[str]:
    return list(CANONICAL_CLASS_NAMES)


def all_alias_folders() -> Dict[str, List[str]]:
    return CLASS_ALIASES.copy()