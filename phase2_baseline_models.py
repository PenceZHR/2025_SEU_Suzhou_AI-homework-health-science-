"""
Phase 2: Baseline Model Construction
=====================================
This module implements lightweight CNN models from scratch for
medical image classification (chest X-ray diagnosis).

Contains:
- Custom CNN architectures (Baseline, Enhanced)
- Data loading and augmentation pipeline
- Training loop with early stopping
- Evaluation metrics and visualization

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, precision_score, recall_score)
import seaborn as sns

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==============================================================================
# DATASET CLASS
# ==============================================================================

class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for medical images.
    
    Handles loading, preprocessing, and augmentation of medical images.
    """
    
    def __init__(self, 
                 data_root: str,
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing class folders
            transform: Optional torchvision transforms
            target_size: Target image size (height, width)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size
        
        self.image_paths = []
        self.labels = []
        self.class_names = {}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Discover and load all images from class folders."""
        class_idx = 0
        
        # Look for class folders
        for folder in sorted(self.data_root.iterdir()):
            if folder.is_dir():
                # Extract class number from folder name
                folder_name = folder.name.lower()
                if 'class' in folder_name:
                    try:
                        class_num = int(''.join(filter(str.isdigit, folder_name)))
                    except ValueError:
                        class_num = class_idx
                else:
                    class_num = class_idx
                
                self.class_names[class_num] = folder.name
                
                # Find all images
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG']:
                    for img_path in folder.glob(ext):
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_num)
                
                class_idx += 1
        
        self.num_classes = len(self.class_names)
        print(f"Loaded {len(self.image_paths)} images from {self.num_classes} classes")
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and preprocess a single image."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Convert to tensor
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==============================================================================
# CNN ARCHITECTURES
# ==============================================================================

class BaselineCNN(nn.Module):
    """
    Baseline CNN for medical image classification.
    
    Architecture:
    - 3 convolutional blocks with batch normalization
    - Global average pooling
    - 2 fully connected layers with dropout
    
    ~500K parameters (lightweight)
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1):
        super(BaselineCNN, self).__init__()
        
        self.model_name = "BaselineCNN"
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (before classifier)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class EnhancedCNN(nn.Module):
    """
    Enhanced CNN with residual connections for better gradient flow.
    
    Architecture:
    - 4 convolutional blocks with skip connections
    - Squeeze-and-Excitation attention
    - Global average pooling
    - 2 fully connected layers
    
    ~1.5M parameters
    """
    
    def __init__(self, num_classes: int = 4, input_channels: int = 1):
        super(EnhancedCNN, self).__init__()
        
        self.model_name = "EnhancedCNN"
        
        # Initial convolution
        self.conv_init = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 224 -> 56
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)
        
        # Squeeze-and-Excitation block
        self.se = SEBlock(256, reduction=16)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block."""
        return ResidualBlock(in_channels, out_channels)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.se(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings."""
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.se(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        out = self.pool(out)
        
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric: float) -> bool:
        if self.best_score is None:
            self.best_score = val_metric
        elif self._is_improvement(val_metric):
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def _is_improvement(self, val_metric: float) -> bool:
        if self.mode == 'min':
            return val_metric < self.best_score - self.min_delta
        else:
            return val_metric > self.best_score + self.min_delta


class Trainer:
    """
    Training pipeline for medical image classification.
    
    Handles:
    - Training loop with validation
    - Metrics tracking
    - Model checkpointing
    - Learning rate scheduling
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[object] = None,
                 device: torch.device = device,
                 output_dir: str = "./training_results"):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int, early_stopping: Optional[EarlyStopping] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs to train
            early_stopping: Optional early stopping callback
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model.model_name}")
        print(f"{'='*60}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_acc, is_best=True)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Early stopping
            if early_stopping and early_stopping(val_loss):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        filename = f"{self.model.model_name}_best.pth" if is_best else f"{self.model.model_name}_epoch{epoch}.pth"
        torch.save(checkpoint, self.output_dir / filename)
    
    def plot_training_curves(self) -> str:
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model.model_name} - Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model.model_name} - Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[2].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'{self.model.model_name} - Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{self.model.model_name}_training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def load_best_model(self):
        """Load the best model state."""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        return self.model


# ==============================================================================
# EVALUATION UTILITIES
# ==============================================================================

class Evaluator:
    """Model evaluation with comprehensive metrics."""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, 
                 class_names: Dict[int, str], device: torch.device = device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        
    def evaluate(self) -> Dict:
        """Run full evaluation on test set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        results = {
            'accuracy': accuracy_score(all_labels, all_preds) * 100,
            'precision_macro': precision_score(all_labels, all_preds, average='macro') * 100,
            'recall_macro': recall_score(all_labels, all_preds, average='macro') * 100,
            'f1_macro': f1_score(all_labels, all_preds, average='macro') * 100,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        # Classification report
        target_names = [self.class_names.get(i, f'Class {i}') for i in sorted(self.class_names.keys())]
        results['classification_report'] = classification_report(
            all_labels, all_preds, target_names=target_names, output_dict=True
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str) -> str:
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        class_labels = [f'Class {i}' for i in sorted(self.class_names.keys())]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {self.model.model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get data transforms for training or validation.
    
    Args:
        mode: 'train' for augmentation, 'val'/'test' for no augmentation
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def get_model_memory(model: nn.Module, input_size: Tuple = (1, 1, 224, 224)) -> float:
    """Estimate model memory usage in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def create_data_loaders(data_root: str, 
                        batch_size: int = 32,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Create full dataset
    full_dataset = MedicalImageDataset(data_root, target_size=(224, 224))
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} images")
    print(f"  Val:   {val_size} images")
    print(f"  Test:  {test_size} images")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_dataset.class_names


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_baseline_training(data_root: str, output_dir: str = "./baseline_results"):
    """
    Run complete baseline model training pipeline.
    """
    print("\n" + "🔬" * 30)
    print("PHASE 2: BASELINE MODEL TRAINING")
    print("🔬" * 30)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    NUM_CLASSES = 4
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_root, batch_size=BATCH_SIZE
    )
    
    # Results storage
    all_results = {}
    
    # Train both models
    models_to_train = [
        ('BaselineCNN', BaselineCNN(num_classes=NUM_CLASSES)),
        ('EnhancedCNN', EnhancedCNN(num_classes=NUM_CLASSES))
    ]
    
    for model_name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=10, mode='min')
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=str(output_dir / model_name)
        )
        
        # Train
        history = trainer.train(epochs=EPOCHS, early_stopping=early_stopping)
        
        # Plot training curves
        trainer.plot_training_curves()
        
        # Load best model and evaluate
        best_model = trainer.load_best_model()
        evaluator = Evaluator(best_model, test_loader, class_names, device)
        eval_results = evaluator.evaluate()
        
        # Plot confusion matrix
        cm_path = output_dir / model_name / f'{model_name}_confusion_matrix.png'
        evaluator.plot_confusion_matrix(eval_results['confusion_matrix'], str(cm_path))
        
        # Store results
        all_results[model_name] = {
            'history': history,
            'test_accuracy': eval_results['accuracy'],
            'test_f1': eval_results['f1_macro'],
            'parameters': count_parameters(model),
            'memory_mb': get_model_memory(model)
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"  Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  F1 Score: {eval_results['f1_macro']:.2f}%")
        print(f"  Parameters: {all_results[model_name]['parameters']['total']:,}")
    
    # Save summary
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, results in all_results.items():
            json_results[model_name] = {
                'test_accuracy': results['test_accuracy'],
                'test_f1': results['test_f1'],
                'parameters': results['parameters'],
                'memory_mb': results['memory_mb'],
                'final_train_loss': results['history']['train_loss'][-1],
                'final_val_loss': results['history']['val_loss'][-1],
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to: {output_dir}")
    
    return all_results


if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "./sample_data"  # Update with actual dataset path
    OUTPUT_DIR = "./baseline_results"
    
    # Check if dataset exists
    if not Path(DATA_ROOT).exists():
        print(f"Dataset not found at {DATA_ROOT}")
        print("Please update DATA_ROOT with the correct path to Dataset2")
    else:
        results = run_baseline_training(DATA_ROOT, OUTPUT_DIR)
