"""
Fine-tune the European-trained model on Indian satellite imagery.

Usage:
  python scripts/finetune_indian.py --data-dir /path/to/labeled/indian/images --epochs 15 --output best_indian_model.pth

Expected folder structure:
  labeled_indian_data/
        Crop/
      image1.png
      image2.png
      ...
    Forest/
    Highway/
        ...etc for all 6 merged classes
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from stages.stage1_single_cell.core.landcover_pipeline import (
    CANONICAL_CLASS_NAMES,
    get_device,
    load_model,
    predict_single_image,
)
from stages.stage2_multiclass.taxonomy.class_taxonomy import CLASS_ALIASES, canonicalize_label


class IndianSatelliteDataset(Dataset):
    """Load satellite images from class folders."""

    def __init__(
        self,
        root_dir: str,
        transforms_aug: transforms.Compose,
        image_extensions: set = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'},
    ):
        self.root_dir = Path(root_dir)
        self.transforms_aug = transforms_aug
        self.image_extensions = image_extensions
        self.images: List[Tuple[str, str]] = []  # (file_path, class_name)
        self.class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_CLASS_NAMES)}

        self._load_images()

        if not self.images:
            raise ValueError(f'No images found in {root_dir}')

    def _load_images(self):
        """Scan class folders and collect image paths."""
        for class_name in CANONICAL_CLASS_NAMES:
            candidate_folders = [class_name] + CLASS_ALIASES.get(class_name, [])
            image_files = []

            for folder_name in candidate_folders:
                class_dir = self.root_dir / folder_name
                if not class_dir.exists():
                    continue

                for ext in self.image_extensions:
                    image_files.extend(class_dir.glob(f'*{ext}'))
                    image_files.extend(class_dir.glob(f'*{ext.upper()}'))

            for img_path in image_files:
                self.images.append((str(img_path), class_name))

            print(f'  {class_name:25s} {len(image_files):4d} images')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_name = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            image = Image.new('RGB', (224, 224), color='gray')

        image = self.transforms_aug(image)
        label = self.class_to_idx[canonicalize_label(class_name) or class_name]
        return image, label


def build_training_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build augmented training and clean validation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def rebuild_classifier_head(model: nn.Module, num_classes: int) -> None:
    """Replace the final classifier layer so fine-tuning outputs merged classes."""
    if hasattr(model, 'network') and hasattr(model.network, 'fc'):
        fc = model.network.fc
        if isinstance(fc, nn.Sequential):
            layers = list(fc.children())
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    in_features = layers[idx].in_features
                    layers[idx] = nn.Linear(in_features, num_classes)
                    model.network.fc = nn.Sequential(*layers)
                    return
        if isinstance(fc, nn.Linear):
            model.network.fc = nn.Sequential(
                nn.Linear(fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
                nn.LogSoftmax(dim=1),
            )
            return

    if hasattr(model, 'fc'):
        fc = model.fc
        if isinstance(fc, nn.Sequential):
            layers = list(fc.children())
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    in_features = layers[idx].in_features
                    layers[idx] = nn.Linear(in_features, num_classes)
                    model.fc = nn.Sequential(*layers)
                    return
        if isinstance(fc, nn.Linear):
            model.fc = nn.Sequential(
                nn.Linear(fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
                nn.LogSoftmax(dim=1),
            )



def setup_finetuning_model(model: nn.Module, unfreeze_blocks: int = 2) -> None:
    """
    Freeze most of the model; unfreeze only the last N blocks + head.

    Args:
        model: Pre-trained model
        unfreeze_blocks: Number of residual blocks to unfreeze from the end
    """
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'network'):
        network = model.network
        trainable_count = 0

        if unfreeze_blocks > 0:
            block_names = [name for name, _ in network.named_modules() if 'layer' in name]
            for name, module in network.named_modules():
                is_final_block = any(name.startswith(f'layer{4 - i}') for i in range(min(unfreeze_blocks, 4)))
                if is_final_block or 'fc' in name:
                    for param in module.parameters():
                        param.requires_grad = True
                        trainable_count += 1

        print(f'Trainable parameters: {trainable_count:,}')
    else:
        for param in model.fc.parameters():
            param.requires_grad = True
        print('Unfroze head layer.')


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.1f}%'})

    return total_loss / len(train_loader), 100 * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc='Validating', unit='batch'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune EuroSAT model on Indian satellite imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python scripts/finetune_indian.py --data-dir labeled_indian_data --epochs 20 --output fine_tuned_model.pth
  
Expected folder structure:
  labeled_indian_data/
        Crop/
    Forest/
    HerbaceousVegetation/
        ... (all 6 merged classes)
        ''',
    )

    parser.add_argument('--data-dir', type=str, default='data/cell_training/labeled_cells', help='Root directory with class folders')
    parser.add_argument('--model', type=str, default='best_eurosat_model.pth', help='Pre-trained model path')
    parser.add_argument('--output', type=str, default='best_indian_model.pth', help='Output fine-tuned model path')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to fine-tune')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--unfreeze-blocks', type=int, default=2, help='Number of residual blocks to unfreeze')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print('=' * 70)
    print('Indian Satellite Fine-tuning')
    print('=' * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f'Device: {device}')

    print('\nLoading pre-trained model...')
    model = load_model(args.model, device=device)
    rebuild_classifier_head(model, num_classes=len(CANONICAL_CLASS_NAMES))

    print('\nLoading dataset...')
    train_transform, val_transform = build_training_transforms()
    full_dataset = IndianSatelliteDataset(args.data_dir, train_transform)

    total_images = len(full_dataset)
    val_size = int(total_images * args.val_split)
    train_size = total_images - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_dataset.dataset.transforms_aug = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'Dataset loaded:')
    print(f'  Training: {len(train_dataset)} images')
    print(f'  Validation: {len(val_dataset)} images')

    print('\nSetting up fine-tuning...')
    setup_finetuning_model(model, unfreeze_blocks=args.unfreeze_blocks)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    no_improve_count = 0
    history = []

    print(f'\nStarting fine-tuning for {args.epochs} epochs...')
    print('=' * 70)

    for epoch in range(args.epochs):
        print(f'\nEpoch [{epoch + 1}/{args.epochs}]')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')

        history.append(
            {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), args.output)
            print(f'âœ… Best model saved! (Val Acc: {val_acc:.1f}%)')
        else:
            no_improve_count += 1
            if no_improve_count >= args.early_stopping_patience:
                print(f'\nâ¹ï¸  Early stopping triggered (no improvement for {args.early_stopping_patience} epochs)')
                break

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.2e}')

    print('\n' + '=' * 70)
    print(f'Fine-tuning complete!')
    print(f'Best Val Accuracy: {best_val_acc:.1f}%')
    print(f'Model saved to: {args.output}')

    history_path = args.output.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to: {history_path}')

    print('=' * 70)


if __name__ == '__main__':
    main()

