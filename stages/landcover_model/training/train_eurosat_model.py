"""
EuroSAT Land Cover Classification - Training Script
Train a Wide ResNet-50-2 model on EuroSAT RGB dataset
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from stages.landcover_model.taxonomy.landcover_class_taxonomy import CANONICAL_CLASS_NAMES, canonicalize_label

# ============================================
# CONFIGURATION
# ============================================
DATA_CSV = 'eurosat_rgb_data.csv'
DATA_DIR = 'data/EuroSAT_RGB/EuroSAT_RGB'
MODEL_SAVE_PATH = 'data/model'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 0.1
VALID_SIZE = 0.1
MAX_EPOCHS_STOP = 3  # Early stopping patience

NUM_CLASSES = len(CANONICAL_CLASS_NAMES)
CLASS_IDX_LABELS = {name: idx for idx, name in enumerate(CANONICAL_CLASS_NAMES)}

# ============================================
# DATASET CLASS
# ============================================
class EuroSATDataset(Dataset):
    """EuroSAT RGB Dataset"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        canonical_label = canonicalize_label(row['label'])
        if canonical_label is None:
            raise ValueError(f"Unsupported label in CSV: {row['label']}")
        label = CLASS_IDX_LABELS[canonical_label]
        return image, label

# ============================================
# MODEL DEFINITION
# ============================================
def accuracy(outputs, labels):
    """Calculate accuracy"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class MulticlassClassifierBase(nn.Module):
    """Base class for classifier with training/validation steps"""
    
    def training_step(self, batch, criterion):
        img, label = batch
        out = self(img)
        loss = criterion(out, label)
        accu = accuracy(out, label)
        return accu, loss
    
    def validation_step(self, batch, criterion):
        img, label = batch
        out = self(img)
        loss = criterion(out, label)
        accu = accuracy(out, label)
        return {"val_loss": loss.detach(), "val_acc": accu}
    
    def validation_epoch_ends(self, outputs):
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

class LULC_Model(MulticlassClassifierBase):
    """Land Use Land Cover Classification Model"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = models.wide_resnet50_2(pretrained=True)
        n_inputs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        """Freeze base layers for transfer learning"""
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        """Unfreeze all layers"""
        for param in self.network.parameters():
            param.requires_grad = True

# ============================================
# DEVICE MANAGEMENT
# ============================================
def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move data to device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrapper to move data to device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)

# ============================================
# TRAINING FUNCTIONS
# ============================================
@torch.no_grad()
def evaluate(model, valid_loader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    outputs = [model.validation_step(batch, criterion) for batch in valid_loader]
    return model.validation_epoch_ends(outputs)

def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, valid_loader, criterion, device,
        weight_decay=0, grad_clip=None, max_epochs_stop=3):
    """Train the model"""
    
    history = []
    valid_loss_min = np.inf
    valid_acc_max = 0
    epochs_no_improve = 0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_accus = []
        lrs = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            accu, loss = model.training_step(batch, criterion)
            train_losses.append(loss)
            train_accus.append(accu)
            
            loss.backward()
            
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accu.item()*100:.2f}%'
            })
        
        # Validation phase
        result = evaluate(model, valid_loader, criterion)
        scheduler.step(result['val_loss'])
        
        # Calculate epoch metrics
        train_loss = torch.stack(train_losses).mean().item()
        train_accu = torch.stack(train_accus).mean().item()
        
        result["train_loss"] = train_loss
        result["train_accu"] = train_accu
        result["lrs"] = lrs
        
        # Print epoch results
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accu*100:.2f}%")
        print(f"  Val Loss: {result['val_loss']:.4f} | Val Acc: {result['val_acc']*100:.2f}%")
        print(f"  Learning Rate: {lrs[-1]:.6f}")
        
        # Save best model based on validation accuracy
        valid_loss = result['val_loss']
        valid_acc = result['val_acc']
        
        if valid_acc > valid_acc_max:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  âœ… New best model saved! (Val Acc: {valid_acc*100:.2f}%)")
            valid_acc_max = valid_acc
            epochs_no_improve = 0
        elif valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= max_epochs_stop:
            print(f"\nâš ï¸  Early stopping after {epoch+1} epochs (no improvement for {max_epochs_stop} epochs)")
            history.append(result)
            break
        
        history.append(result)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Val Loss: {valid_loss_min:.4f}")
    print(f"Best Val Acc: {valid_acc_max*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")
    
    return history

# ============================================
# MAIN FUNCTION
# ============================================
def main():
    """Main training pipeline"""
    
    print("="*60)
    print("EuroSAT Land Cover Classification Training")
    print("="*60)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("âš ï¸  Training on CPU - this will take 2-4 hours")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\nLoading dataset from: {DATA_CSV}")
    full_dataset = EuroSATDataset(DATA_CSV, DATA_DIR, transform=transform)
    print(f"Total images: {len(full_dataset)}")
    
    # Split train/validation
    val_size = int(VALID_SIZE * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Move to device
    train_loader = DeviceDataLoader(train_loader, device)
    valid_loader = DeviceDataLoader(valid_loader, device)
    
    # Create model
    print("\nCreating model...")
    model = LULC_Model(num_classes=NUM_CLASSES)
    model = to_device(model, device)
    print("âœ… Model created (Wide ResNet-50-2 with transfer learning)")
    
    # Loss function
    criterion = nn.NLLLoss()  # Negative Log-Likelihood for LogSoftmax output
    
    # Train
    history = fit(
        epochs=EPOCHS,
        max_lr=LEARNING_RATE,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        max_epochs_stop=MAX_EPOCHS_STOP
    )
    
    print("\nâœ… Training finished! You can now use the model for predictions.")
    print(f"Model weights saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()

