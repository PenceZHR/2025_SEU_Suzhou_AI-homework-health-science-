"""
Optimized Medical Image Classification Pipeline
================================================
Enhanced training strategies to achieve >98% accuracy

Key Improvements:
1. Better data augmentation
2. Larger image size (299x299)
3. Label smoothing
4. Mixup augmentation
5. Cosine annealing with warm restarts
6. Test-time augmentation (TTA)
7. Ensemble methods
8. Optimized hyperparameters

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

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==============================================================================
# ENHANCED DATA AUGMENTATION
# ==============================================================================

class MedicalImageDatasetOptimized(Dataset):
    """
    Optimized dataset with strong augmentation for medical images.
    """
    
    def __init__(self, data_root: str, mode: str = 'train', target_size: Tuple[int, int] = (299, 299)):
        self.data_root = Path(data_root)
        self.mode = mode
        self.target_size = target_size
        self.image_paths = []
        self.labels = []
        self.class_names = {}
        self._load_dataset()
        
        # Define augmentation transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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
        
        # Load image
        image = Image.open(img_path).convert('L')
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Apply augmentation
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize with ImageNet-like statistics (adjusted for grayscale)
        mean = 0.485
        std = 0.229
        image = (image - mean) / std
        
        image = torch.from_numpy(image).unsqueeze(0)
        
        return image, label


# ==============================================================================
# MIXUP AUGMENTATION
# ==============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# LABEL SMOOTHING LOSS
# ==============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # Smooth labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_preds)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        
        loss = (-smooth_labels * log_preds).sum(dim=-1).mean()
        return loss


# ==============================================================================
# OPTIMIZED MODEL ARCHITECTURES
# ==============================================================================

class OptimizedResNet50(nn.Module):
    """ResNet50 with optimizations for medical imaging."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.model_name = "ResNet50_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained ResNet50 weights...")
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                self.backbone = models.resnet50(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.resnet50(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.resnet50(weights=None)
            pretrained = False
        
        # Modify first conv for grayscale
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Enhanced classifier with more capacity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


class OptimizedEfficientNetB2(nn.Module):
    """EfficientNet-B2 (larger than B0) for better accuracy."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.model_name = "EfficientNetB2_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained EfficientNet-B2 weights...")
                weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
                self.backbone = models.efficientnet_b2(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.efficientnet_b2(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.efficientnet_b2(weights=None)
            pretrained = False
        
        # Modify first conv for grayscale
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Enhanced classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


class OptimizedDenseNet169(nn.Module):
    """DenseNet169 (larger than 121) for better accuracy."""
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        self.model_name = "DenseNet169_Optimized"
        
        try:
            if pretrained:
                print(f"  Loading pretrained DenseNet169 weights...")
                weights = models.DenseNet169_Weights.IMAGENET1K_V1
                self.backbone = models.densenet169(weights=weights)
                print(f"  ✓ Pretrained weights loaded")
            else:
                self.backbone = models.densenet169(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not load pretrained weights: {e}")
            self.backbone = models.densenet169(weights=None)
            pretrained = False
        
        # Modify first conv for grayscale
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Enhanced classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(out)
        return out
    
    def get_features(self, x):
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)


# ==============================================================================
# OPTIMIZED TRAINER
# ==============================================================================

class OptimizedTrainer:
    """
    Optimized training pipeline with advanced techniques.
    """
    
    def __init__(self, model, train_loader, val_loader, num_classes, output_dir, 
                 use_mixup=True, use_label_smoothing=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mixup = use_mixup
        
        # Loss function with label smoothing
        if use_label_smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4,  # Lower initial LR
            weight_decay=1e-3  # Stronger regularization
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.best_val_acc = 0
        self.best_model_state = None
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            
            # Apply mixup with 50% probability
            use_mixup_this_batch = self.use_mixup and np.random.random() > 0.5
            
            if use_mixup_this_batch:
                data, target_a, target_b, lam = mixup_data(data, target, alpha=0.4)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target_a.size(0)
                # For mixup, calculate weighted accuracy
                correct += (lam * predicted.eq(target_a).sum().item() + 
                           (1 - lam) * predicted.eq(target_b).sum().item())
            else:
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
        
        self.scheduler.step()
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = F.cross_entropy(output, target)  # Use standard CE for validation
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, epochs=100, patience=20):
        print(f"\n{'='*70}")
        print(f"Training {self.model.model_name}")
        print(f"{'='*70}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {device}")
        print(f"Mixup: {self.use_mixup}")
        print(f"{'='*70}\n")
        
        no_improve = 0
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                no_improve = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'val_acc': val_acc,
                    'history': self.history
                }, self.output_dir / f'{self.model.model_name}_best.pth')
                
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ★ | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            else:
                no_improve += 1
                if epoch % 5 == 0 or epoch <= 10:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                          f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        print(f"\n{'='*70}")
        print(f"Training Complete! Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*70}")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{self.model.model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[1].axhline(y=98, color='g', linestyle='--', label='Target (98%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{self.model.model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[2].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'{self.model.model_name} - Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{self.model.model_name}_training_curves.png', dpi=150)
        plt.close()


# ==============================================================================
# TEST-TIME AUGMENTATION (TTA)
# ==============================================================================

def test_time_augmentation(model, image, num_augments=10):
    """
    Apply test-time augmentation for better predictions.
    """
    model.eval()
    
    predictions = []
    
    # Original prediction
    with torch.no_grad():
        pred = F.softmax(model(image), dim=1)
        predictions.append(pred)
    
    # Augmented predictions
    augment_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
        lambda x: torch.flip(x, dims=[2]),  # Vertical flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90 degree rotation
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # 180 degree rotation
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270 degree rotation
    ]
    
    for transform in augment_transforms[1:num_augments]:
        with torch.no_grad():
            augmented = transform(image)
            pred = F.softmax(model(augmented), dim=1)
            predictions.append(pred)
    
    # Average predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return avg_prediction


def evaluate_with_tta(model, test_loader, num_augments=5):
    """Evaluate model with test-time augmentation."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating with Test-Time Augmentation...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            
            # TTA prediction
            avg_pred = test_time_augmentation(model, data, num_augments)
            _, predicted = avg_pred.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return accuracy, f1, np.array(all_preds), np.array(all_labels)


# ==============================================================================
# ENSEMBLE MODEL
# ==============================================================================

class EnsembleModel:
    """Ensemble of multiple models for better predictions."""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.model_name = "Ensemble"
    
    def predict(self, x):
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred
    
    def evaluate(self, test_loader):
        all_preds = []
        all_labels = []
        
        print("Evaluating Ensemble Model...")
        
        for data, target in test_loader:
            data = data.to(device)
            
            avg_pred = self.predict(data)
            _, predicted = avg_pred.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        return accuracy, f1, np.array(all_preds), np.array(all_labels)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def create_optimized_loaders(data_root: str, batch_size: int = 32, 
                              train_ratio: float = 0.8, val_ratio: float = 0.1,
                              image_size: int = 299):
    """Create data loaders with class balancing."""
    
    # Create datasets
    full_dataset = MedicalImageDatasetOptimized(data_root, mode='train', target_size=(image_size, image_size))
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create class-balanced sampler for training
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"\nDataset split (Image size: {image_size}x{image_size}):")
    print(f"  Train: {train_size} images")
    print(f"  Val:   {val_size} images")
    print(f"  Test:  {test_size} images")
    print(f"  Class distribution: {dict(enumerate(class_counts))}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_dataset.class_names, full_dataset.num_classes


def run_optimized_training(data_root: str, output_dir: str = "./optimized_results"):
    """Run optimized training pipeline."""
    
    print("\n" + "🚀" * 35)
    print("OPTIMIZED TRAINING PIPELINE FOR >98% ACCURACY")
    print("🚀" * 35)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders with larger image size
    train_loader, val_loader, test_loader, class_names, num_classes = create_optimized_loaders(
        data_root, batch_size=16, image_size=299  # Larger images, smaller batch
    )
    
    # Models to train
    models_to_train = [
        ('ResNet50', OptimizedResNet50(num_classes=num_classes, pretrained=True, dropout=0.5)),
        ('EfficientNetB2', OptimizedEfficientNetB2(num_classes=num_classes, pretrained=True, dropout=0.4)),
        ('DenseNet169', OptimizedDenseNet169(num_classes=num_classes, pretrained=True, dropout=0.4)),
    ]
    
    trained_models = []
    all_results = {}
    
    for model_name, model in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            output_dir=str(output_dir / model_name),
            use_mixup=True,
            use_label_smoothing=True
        )
        
        history = trainer.train(epochs=100, patience=25)
        trainer.plot_history()
        
        # Evaluate with TTA
        tta_acc, tta_f1, preds, labels = evaluate_with_tta(model, test_loader, num_augments=5)
        
        print(f"\n{model_name} Test Results (with TTA):")
        print(f"  Accuracy: {tta_acc:.2f}%")
        print(f"  F1 Score: {tta_f1:.2f}%")
        
        all_results[model_name] = {
            'val_acc': trainer.best_val_acc,
            'test_acc_tta': tta_acc,
            'test_f1_tta': tta_f1,
            'history': history
        }
        
        trained_models.append(model)
    
    # Ensemble evaluation
    print(f"\n{'='*70}")
    print("ENSEMBLE MODEL EVALUATION")
    print(f"{'='*70}")
    
    ensemble = EnsembleModel(trained_models)
    ens_acc, ens_f1, ens_preds, ens_labels = ensemble.evaluate(test_loader)
    
    print(f"\nEnsemble Results:")
    print(f"  Accuracy: {ens_acc:.2f}%")
    print(f"  F1 Score: {ens_f1:.2f}%")
    
    all_results['Ensemble'] = {
        'test_acc': ens_acc,
        'test_f1': ens_f1
    }
    
    # Save summary
    summary_path = output_dir / 'optimized_results_summary.json'
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(summary_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Val Acc':<12} {'Test Acc (TTA)':<15} {'Test F1':<12}")
    print("-" * 60)
    
    for name, results in all_results.items():
        val_acc = results.get('val_acc', results.get('test_acc', 0))
        test_acc = results.get('test_acc_tta', results.get('test_acc', 0))
        test_f1 = results.get('test_f1_tta', results.get('test_f1', 0))
        print(f"{name:<20} {val_acc:.2f}%{'':<6} {test_acc:.2f}%{'':<9} {test_f1:.2f}%")
    
    print(f"{'='*70}")
    print(f"\n✅ Results saved to: {output_dir}")
    
    return all_results, trained_models


if __name__ == "__main__":
    DATA_ROOT = "./Dataset2"  # Update this path
    OUTPUT_DIR = "./optimized_results"
    
    if not Path(DATA_ROOT).exists():
        print(f"Dataset not found at {DATA_ROOT}")
        print("Please update DATA_ROOT with the correct path")
    else:
        results, models = run_optimized_training(DATA_ROOT, OUTPUT_DIR)
