"""
CPU-Friendly Medical Image Classification Pipeline
===================================================
Optimized for faster training on CPU while still achieving good accuracy.

Changes from full version:
- Smaller image size (224x224)
- Lighter models (ResNet18, EfficientNet-B0, DenseNet121)
- Smaller batch processing
- Fewer epochs with progress display
- Option to train single model

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import random

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cpu':
    print("⚠️  WARNING: Running on CPU. Training will be slow.")
    print("   For faster training, please install CUDA-enabled PyTorch.")


# ==============================================================================
# DATASET
# ==============================================================================

class MedicalImageDataset(Dataset):
    """Dataset with augmentation for medical images."""
    
    def __init__(self, data_root: str, mode: str = 'train', target_size: Tuple[int, int] = (224, 224)):
        self.data_root = Path(data_root)
        self.mode = mode
        self.target_size = target_size
        self.image_paths = []
        self.labels = []
        self.class_names = {}
        self._load_dataset()
        
        # Augmentation for training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            self.transform = None
    
    def _load_dataset(self):
        for folder in sorted(self.data_root.iterdir()):
            if folder.is_dir():
                folder_name = folder.name.lower()
                if 'class' in folder_name:
                    try:
                        class_num = int(''.join(filter(str.isdigit, folder_name)))
                    except:
                        class_num = len(self.class_names)
                else:
                    class_num = len(self.class_names)
                
                self.class_names[class_num] = folder.name
                
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG', '*.JPEG']:
                    for img_path in folder.glob(ext):
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_num)
        
        self.num_classes = len(self.class_names)
        print(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('L')
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
        
        image = np.array(image, dtype=np.float32) / 255.0
        mean, std = 0.485, 0.229
        image = (image - mean) / std
        image = torch.from_numpy(image).unsqueeze(0)
        
        return image, label


# ==============================================================================
# LABEL SMOOTHING
# ==============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_preds)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return (-smooth_labels * log_preds).sum(dim=-1).mean()


# ==============================================================================
# MODELS (Lighter versions)
# ==============================================================================

class OptimizedResNet18(nn.Module):
    """ResNet18 - lighter and faster than ResNet50."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.model_name = "ResNet18_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained ResNet18 weights...")
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.resnet18(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.resnet18(weights=None)
            pretrained = False
        
        # Modify for grayscale
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class OptimizedEfficientNetB0(nn.Module):
    """EfficientNet-B0 - efficient and accurate."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.model_name = "EfficientNetB0_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained EfficientNet-B0 weights...")
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                self.backbone = models.efficientnet_b0(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.efficientnet_b0(weights=None)
            pretrained = False
        
        # Modify for grayscale
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class OptimizedDenseNet121(nn.Module):
    """DenseNet121 - good for medical imaging."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.model_name = "DenseNet121_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained DenseNet121 weights...")
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.backbone = models.densenet121(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.densenet121(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.densenet121(weights=None)
            pretrained = False
        
        # Modify for grayscale
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==============================================================================
# TRAINER WITH PROGRESS
# ==============================================================================

class Trainer:
    """Training pipeline with detailed progress display."""
    
    def __init__(self, model, train_loader, val_loader, num_classes, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0
        self.best_model_state = None
    
    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        num_batches = len(self.train_loader)
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress display every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                current_acc = 100. * correct / total
                print(f"\r  Epoch {epoch}/{total_epochs} | Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {total_loss/(batch_idx+1):.4f} | Acc: {current_acc:.2f}% | "
                      f"ETA: {eta:.0f}s", end='', flush=True)
        
        print()  # New line after epoch
        self.scheduler.step()
        
        return total_loss / num_batches, 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs=50, patience=15):
        print(f"\n{'='*70}")
        print(f"Training {self.model.model_name}")
        print(f"{'='*70}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            val_loss, val_acc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                no_improve = 0
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'val_acc': val_acc
                }, self.output_dir / f'{self.model.model_name}_best.pth')
                print(f"  ➜ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ★ NEW BEST!")
            else:
                no_improve += 1
                print(f"  ➜ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if no_improve >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete! Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*70}")
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                output = self.model(data)
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        return accuracy, f1, np.array(all_preds), np.array(all_labels)
    
    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model.model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val')
        axes[1].axhline(y=98, color='g', linestyle='--', alpha=0.5, label='98% Target')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model.model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.model.model_name}_curves.png', dpi=150)
        plt.close()
        print(f"  Saved training curves to {self.output_dir / f'{self.model.model_name}_curves.png'}")


# ==============================================================================
# ENSEMBLE
# ==============================================================================

class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def evaluate(self, test_loader):
        all_preds = []
        all_labels = []
        
        for data, target in test_loader:
            data = data.to(device)
            
            # Average predictions from all models
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model(data), dim=1)
                    predictions.append(pred)
            
            avg_pred = torch.stack(predictions).mean(dim=0)
            _, predicted = avg_pred.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        return accuracy, f1


# ==============================================================================
# MAIN
# ==============================================================================

def create_loaders(data_root, batch_size=32, train_ratio=0.8, val_ratio=0.1, image_size=224):
    full_dataset = MedicalImageDataset(data_root, mode='train', target_size=(image_size, image_size))
    
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Class balancing
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"\nDataset split (Image size: {image_size}x{image_size}):")
    print(f"  Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, full_dataset.class_names, full_dataset.num_classes


def run_training(data_root: str, output_dir: str = "./cpu_optimized_results", 
                 model_choice: str = "all", epochs: int = 50):
    """
    Run training pipeline.
    
    Args:
        data_root: Path to dataset
        output_dir: Output directory
        model_choice: "resnet", "efficientnet", "densenet", or "all"
        epochs: Number of training epochs
    """
    print("\n" + "🏥" * 35)
    print("MEDICAL IMAGE CLASSIFICATION - CPU OPTIMIZED")
    print("🏥" * 35)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust batch size based on device
    batch_size = 32 if device.type == 'cuda' else 16
    
    train_loader, val_loader, test_loader, class_names, num_classes = create_loaders(
        data_root, batch_size=batch_size, image_size=224
    )
    
    # Select models to train
    all_models = {
        'resnet': ('ResNet18', OptimizedResNet18(num_classes=num_classes, pretrained=True)),
        'efficientnet': ('EfficientNetB0', OptimizedEfficientNetB0(num_classes=num_classes, pretrained=True)),
        'densenet': ('DenseNet121', OptimizedDenseNet121(num_classes=num_classes, pretrained=True)),
    }
    
    if model_choice == "all":
        models_to_train = list(all_models.values())
    else:
        if model_choice in all_models:
            models_to_train = [all_models[model_choice]]
        else:
            print(f"Unknown model: {model_choice}. Using all models.")
            models_to_train = list(all_models.values())
    
    trained_models = []
    all_results = {}
    
    for model_name, model in models_to_train:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            output_dir=str(output_dir / model_name)
        )
        
        history = trainer.train(epochs=epochs, patience=15)
        trainer.plot_history()
        
        # Evaluate
        test_acc, test_f1, preds, labels = trainer.evaluate(test_loader)
        
        print(f"\n📊 {model_name} Test Results:")
        print(f"   Accuracy: {test_acc:.2f}%")
        print(f"   F1 Score: {test_f1:.2f}%")
        
        all_results[model_name] = {
            'val_acc': trainer.best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1
        }
        
        trained_models.append(model)
    
    # Ensemble (if multiple models)
    if len(trained_models) > 1:
        print(f"\n{'='*70}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*70}")
        
        ensemble = EnsembleModel(trained_models)
        ens_acc, ens_f1 = ensemble.evaluate(test_loader)
        
        print(f"\n📊 Ensemble Results:")
        print(f"   Accuracy: {ens_acc:.2f}%")
        print(f"   F1 Score: {ens_f1:.2f}%")
        
        all_results['Ensemble'] = {'test_acc': ens_acc, 'test_f1': ens_f1}
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 60)
    for name, res in all_results.items():
        val_acc = res.get('val_acc', '-')
        test_acc = res.get('test_acc', 0)
        test_f1 = res.get('test_f1', 0)
        val_str = f"{val_acc:.2f}%" if isinstance(val_acc, float) else val_acc
        print(f"{name:<25} {val_str:<12} {test_acc:.2f}%{'':<6} {test_f1:.2f}%")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical Image Classification')
    parser.add_argument('--data_root', type=str, default='./Dataset2', help='Dataset path')
    parser.add_argument('--output_dir', type=str, default='./results_optimized', help='Output directory')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['resnet', 'efficientnet', 'densenet', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    
    args = parser.parse_args()
    
    if not Path(args.data_root).exists():
        print(f"Dataset not found at {args.data_root}")
    else:
        run_training(args.data_root, args.output_dir, args.model, args.epochs)
