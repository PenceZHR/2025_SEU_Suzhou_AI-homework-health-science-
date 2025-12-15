"""
Phase 3: Transfer Learning with Pre-trained Models
===================================================
This module implements transfer learning approaches using mature
pre-trained models for medical image classification.

Models included:
- ResNet18 (Microsoft Research)
- EfficientNet-B0 (Google)
- DenseNet121 (Cornell/Facebook)

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
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Import from phase 2
from phase2_baseline_models import (
    MedicalImageDataset, Trainer, Evaluator, EarlyStopping,
    count_parameters, get_model_memory, create_data_loaders, device, SEED
)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==============================================================================
# TRANSFER LEARNING MODEL WRAPPERS
# ==============================================================================

class TransferResNet18(nn.Module):
    """
    ResNet18 adapted for grayscale medical image classification.
    
    Features:
    - Modified first conv layer for single-channel input
    - Pretrained ImageNet weights (where applicable)
    - Custom classifier head
    
    Original paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = False):
        super(TransferResNet18, self).__init__()
        
        self.model_name = "ResNet18"
        
        # Try to load pretrained weights, fall back to random init if download fails
        try:
            if pretrained:
                print(f"  Loading pretrained ResNet18 weights...")
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
                print(f"  ✓ Pretrained weights loaded successfully")
            else:
                self.backbone = models.resnet18(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not download pretrained weights: {e}")
            print(f"  → Using random initialization instead")
            self.backbone = models.resnet18(weights=None)
            pretrained = False
        
        # Modify first conv layer for single-channel input
        # Average the weights across RGB channels
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        if pretrained:
            # Initialize with averaged RGB weights
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class TransferEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 adapted for grayscale medical image classification.
    
    Features:
    - Compound scaling for optimal width/depth/resolution
    - Mobile inverted bottleneck convolutions (MBConv)
    - Squeeze-and-excitation optimization
    
    Original paper: "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = False):
        super(TransferEfficientNetB0, self).__init__()
        
        self.model_name = "EfficientNet-B0"
        
        # Try to load pretrained weights, fall back to random init if download fails
        try:
            if pretrained:
                print(f"  Loading pretrained EfficientNet-B0 weights...")
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                self.backbone = models.efficientnet_b0(weights=weights)
                print(f"  ✓ Pretrained weights loaded successfully")
            else:
                self.backbone = models.efficientnet_b0(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not download pretrained weights: {e}")
            print(f"  → Using random initialization instead")
            self.backbone = models.efficientnet_b0(weights=None)
            pretrained = False
        
        # Modify first conv layer for single-channel input
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class TransferDenseNet121(nn.Module):
    """
    DenseNet121 adapted for grayscale medical image classification.
    
    Features:
    - Dense connectivity pattern
    - Feature reuse through concatenation
    - Efficient parameter usage
    
    Original paper: "Densely Connected Convolutional Networks" (Huang et al., 2016)
    Widely used in medical imaging (CheXNet, etc.)
    """
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = False):
        super(TransferDenseNet121, self).__init__()
        
        self.model_name = "DenseNet121"
        
        # Try to load pretrained weights, fall back to random init if download fails
        try:
            if pretrained:
                print(f"  Loading pretrained DenseNet121 weights...")
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
                self.backbone = models.densenet121(weights=weights)
                print(f"  ✓ Pretrained weights loaded successfully")
            else:
                self.backbone = models.densenet121(weights=None)
        except Exception as e:
            print(f"  ⚠ Could not download pretrained weights: {e}")
            print(f"  → Using random initialization instead")
            self.backbone = models.densenet121(weights=None)
            pretrained = False
        
        # Modify first conv layer for single-channel input
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        if pretrained:
            with torch.no_grad():
                self.backbone.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier."""
        features = self.backbone.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


# ==============================================================================
# MODIFIED DATASET FOR TRANSFER LEARNING
# ==============================================================================

class MedicalImageDatasetRGB(MedicalImageDataset):
    """
    Medical image dataset that outputs 3-channel images.
    Converts grayscale to RGB by replicating channels.
    """
    
    def __init__(self, data_root: str, transform=None, target_size=(224, 224)):
        super().__init__(data_root, transform, target_size)
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert to grayscale first
        image = Image.open(img_path).convert('L')
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Convert to numpy and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Single channel tensor
        image = torch.from_numpy(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


from PIL import Image

def create_transfer_data_loaders(data_root: str,
                                  batch_size: int = 32,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  num_workers: int = 4):
    """Create data loaders for transfer learning (single-channel)."""
    return create_data_loaders(data_root, batch_size, train_ratio, val_ratio, num_workers)


# ==============================================================================
# FINE-TUNING STRATEGIES
# ==============================================================================

class GradualUnfreezing:
    """
    Gradually unfreeze layers during training for better fine-tuning.
    
    Strategy:
    - Start with frozen backbone
    - Gradually unfreeze from top layers to bottom
    """
    
    def __init__(self, model: nn.Module, unfreeze_schedule: Dict[int, List[str]]):
        """
        Args:
            model: The model to apply unfreezing to
            unfreeze_schedule: Dict mapping epoch -> list of layer names to unfreeze
        """
        self.model = model
        self.schedule = unfreeze_schedule
        
    def step(self, epoch: int):
        """Unfreeze layers according to schedule."""
        if epoch in self.schedule:
            layers_to_unfreeze = self.schedule[epoch]
            for name, param in self.model.named_parameters():
                for layer_name in layers_to_unfreeze:
                    if layer_name in name:
                        param.requires_grad = True
                        print(f"  Unfreezing: {name}")


class DifferentialLearningRate:
    """
    Apply different learning rates to different parts of the model.
    
    Typically: lower LR for pretrained backbone, higher for new classifier.
    """
    
    @staticmethod
    def get_param_groups(model: nn.Module, 
                         backbone_lr: float = 1e-5,
                         classifier_lr: float = 1e-3) -> List[Dict]:
        """
        Get parameter groups with differential learning rates.
        """
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ]


# ==============================================================================
# COMPARISON UTILITIES
# ==============================================================================

class ModelComparison:
    """
    Compare multiple models on various metrics.
    """
    
    def __init__(self, output_dir: str = "./comparison_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def add_result(self, model_name: str, results: Dict):
        """Add model results for comparison."""
        self.results[model_name] = results
        
    def plot_accuracy_comparison(self) -> str:
        """Plot accuracy comparison across models."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        train_accs = [self.results[m].get('final_train_acc', 0) for m in models]
        val_accs = [self.results[m].get('final_val_acc', 0) for m in models]
        test_accs = [self.results[m].get('test_accuracy', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, train_accs, width, label='Train Acc', color='#3498db')
        bars2 = ax.bar(x, val_accs, width, label='Val Acc', color='#2ecc71')
        bars3 = ax.bar(x + width, test_accs, width, label='Test Acc', color='#e74c3c')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = self.output_dir / 'accuracy_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_parameter_comparison(self) -> str:
        """Plot parameter count comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        params = [self.results[m].get('parameters', {}).get('total', 0) / 1e6 for m in models]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.bar(models, params, color=colors, edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Parameters (Millions)', fontsize=12)
        ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        
        for bar, param in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{param:.2f}M', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / 'parameter_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_training_dynamics(self) -> str:
        """Plot training curves for all models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            history = results.get('history', {})
            if history:
                epochs = range(1, len(history.get('train_loss', [])) + 1)
                
                # Loss
                axes[0].plot(epochs, history.get('train_loss', []), 
                            color=colors[idx], linestyle='-', label=f'{model_name} (train)')
                axes[0].plot(epochs, history.get('val_loss', []),
                            color=colors[idx], linestyle='--', alpha=0.7)
                
                # Accuracy
                axes[1].plot(epochs, history.get('train_acc', []),
                            color=colors[idx], linestyle='-', label=f'{model_name}')
                axes[1].plot(epochs, history.get('val_acc', []),
                            color=colors[idx], linestyle='--', alpha=0.7)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Comparison')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy Comparison')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'training_dynamics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_comparison_table(self) -> str:
        """Generate a comparison table as text."""
        table_lines = []
        table_lines.append("=" * 100)
        table_lines.append("MODEL COMPARISON SUMMARY")
        table_lines.append("=" * 100)
        table_lines.append(f"{'Model':<20} {'Test Acc':<12} {'F1 Score':<12} {'Params':<15} {'Memory (MB)':<12} {'Train Time':<12}")
        table_lines.append("-" * 100)
        
        for model_name, results in self.results.items():
            test_acc = results.get('test_accuracy', 0)
            f1 = results.get('test_f1', 0)
            params = results.get('parameters', {}).get('total', 0)
            memory = results.get('memory_mb', 0)
            train_time = results.get('train_time_min', 0)
            
            table_lines.append(
                f"{model_name:<20} {test_acc:.2f}%{'':<6} {f1:.2f}%{'':<6} {params:,}{'':<5} {memory:.2f}{'':<8} {train_time:.1f} min"
            )
        
        table_lines.append("=" * 100)
        
        table_text = "\n".join(table_lines)
        
        save_path = self.output_dir / 'comparison_table.txt'
        with open(save_path, 'w') as f:
            f.write(table_text)
        
        print(table_text)
        return str(save_path)


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def run_transfer_learning(data_root: str, output_dir: str = "./transfer_results"):
    """
    Run complete transfer learning pipeline with all models.
    """
    print("\n" + "🧠" * 30)
    print("PHASE 3: TRANSFER LEARNING")
    print("🧠" * 30)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 30
    NUM_CLASSES = 4
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_transfer_data_loaders(
        data_root, batch_size=BATCH_SIZE
    )
    
    # Initialize comparison
    comparison = ModelComparison(output_dir=str(output_dir / 'comparison'))
    
    # Models to train
    transfer_models = [
        ('ResNet18', TransferResNet18(num_classes=NUM_CLASSES, pretrained=True)),
        ('EfficientNet-B0', TransferEfficientNetB0(num_classes=NUM_CLASSES, pretrained=True)),
        ('DenseNet121', TransferDenseNet121(num_classes=NUM_CLASSES, pretrained=True)),
    ]
    
    all_results = {}
    
    for model_name, model in transfer_models:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Use differential learning rates
        param_groups = DifferentialLearningRate.get_param_groups(
            model, backbone_lr=1e-5, classifier_lr=1e-3
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        early_stopping = EarlyStopping(patience=8, mode='min')
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=str(output_dir / model_name.replace('-', '_'))
        )
        
        # Train
        history = trainer.train(epochs=EPOCHS, early_stopping=early_stopping)
        train_time = (time.time() - start_time) / 60
        
        # Plot training curves
        trainer.plot_training_curves()
        
        # Evaluate
        best_model = trainer.load_best_model()
        evaluator = Evaluator(best_model, test_loader, class_names, device)
        eval_results = evaluator.evaluate()
        
        # Plot confusion matrix
        cm_path = output_dir / model_name.replace('-', '_') / f'{model_name}_confusion_matrix.png'
        evaluator.plot_confusion_matrix(eval_results['confusion_matrix'], str(cm_path))
        
        # Store results
        results = {
            'history': history,
            'test_accuracy': eval_results['accuracy'],
            'test_f1': eval_results['f1_macro'],
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'parameters': count_parameters(model),
            'memory_mb': get_model_memory(model),
            'train_time_min': train_time
        }
        
        all_results[model_name] = results
        comparison.add_result(model_name, results)
        
        print(f"\n{model_name} Results:")
        print(f"  Test Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  Test F1 Score: {eval_results['f1_macro']:.2f}%")
        print(f"  Parameters: {results['parameters']['total']:,}")
        print(f"  Training Time: {train_time:.1f} minutes")
    
    # Generate comparison visualizations
    print("\n" + "=" * 60)
    print("Generating comparison visualizations...")
    print("=" * 60)
    
    comparison.plot_accuracy_comparison()
    comparison.plot_parameter_comparison()
    comparison.plot_training_dynamics()
    comparison.generate_comparison_table()
    
    # Save summary
    summary_path = output_dir / 'transfer_learning_summary.json'
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {
            'test_accuracy': results['test_accuracy'],
            'test_f1': results['test_f1'],
            'parameters': results['parameters'],
            'memory_mb': results['memory_mb'],
            'train_time_min': results['train_time_min']
        }
    
    with open(summary_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Transfer learning complete! Results saved to: {output_dir}")
    
    return all_results, comparison


if __name__ == "__main__":
    DATA_ROOT = "./sample_data"
    OUTPUT_DIR = "./transfer_results"
    
    if not Path(DATA_ROOT).exists():
        print(f"Dataset not found at {DATA_ROOT}")
        print("Please update DATA_ROOT with the correct path")
    else:
        results, comparison = run_transfer_learning(DATA_ROOT, OUTPUT_DIR)
