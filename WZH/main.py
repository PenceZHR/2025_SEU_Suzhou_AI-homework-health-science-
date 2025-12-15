"""
Medical Image Classification - Main Pipeline
=============================================
Complete pipeline for automated chest X-ray diagnosis using deep learning.

This script orchestrates all phases:
1. Data Exploration
2. Baseline Model Training
3. Transfer Learning
4. Result Analysis

Usage:
    python main.py --data_root Dataset2 --output_dir results --phases 1,2,3,4
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 50
    print("\n" + char * width)
    print(f" {text}")
    print(char * width + "\n")


def print_system_info():
    """Print system and configuration information."""
    print_banner("SYSTEM INFORMATION")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"Random Seed: {SEED}")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_phase1(data_root: str, output_dir: str) -> dict:
    """Run Phase 1: Data Exploration."""
    print_banner("PHASE 1: DATA EXPLORATION", "🔍")
    
    from phase1_data_exploration import MedicalDataExplorer
    
    explorer = MedicalDataExplorer(
        data_root=data_root,
        output_dir=os.path.join(output_dir, "phase1_exploration")
    )
    
    # Execute exploration pipeline
    class_counts = explorer.discover_dataset()
    
    if not class_counts:
        print("ERROR: No images found in dataset!")
        return None
    
    statistics = explorer.analyze_image_properties(sample_size=100)
    explorer.visualize_sample_images(samples_per_class=4)
    explorer.visualize_class_distribution()
    explorer.visualize_intensity_distributions(sample_size=200)
    explorer.generate_exploration_report()
    
    print("✅ Phase 1 Complete!")
    
    return {
        'class_counts': class_counts,
        'statistics': statistics
    }


def run_phase2(data_root: str, output_dir: str) -> dict:
    """Run Phase 2: Baseline Model Training."""
    print_banner("PHASE 2: BASELINE MODEL TRAINING", "🔬")
    
    from phase2_baseline_models import (
        BaselineCNN, EnhancedCNN, Trainer, Evaluator, EarlyStopping,
        create_data_loaders, count_parameters, get_model_memory
    )
    
    phase2_dir = os.path.join(output_dir, "phase2_baseline")
    os.makedirs(phase2_dir, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    NUM_CLASSES = 4
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_root, batch_size=BATCH_SIZE
    )
    
    results = {}
    
    # Models to train
    models = [
        ('BaselineCNN', BaselineCNN(num_classes=NUM_CLASSES)),
        ('EnhancedCNN', EnhancedCNN(num_classes=NUM_CLASSES))
    ]
    
    for model_name, model in models:
        print(f"\n--- Training {model_name} ---")
        
        start_time = time.time()

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=10, mode='min')
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=os.path.join(phase2_dir, model_name)
        )
        
        history = trainer.train(epochs=EPOCHS, early_stopping=early_stopping)
        train_time = (time.time() - start_time) / 60
        
        trainer.plot_training_curves()
        
        # Evaluate
        best_model = trainer.load_best_model()
        evaluator = Evaluator(best_model, test_loader, class_names, device)
        eval_results = evaluator.evaluate()
        
        cm_path = os.path.join(phase2_dir, model_name, f'{model_name}_confusion_matrix.png')
        evaluator.plot_confusion_matrix(eval_results['confusion_matrix'], cm_path)
        
        results[model_name] = {
            'history': history,
            'test_accuracy': eval_results['accuracy'],
            'test_f1': eval_results['f1_macro'],
            'parameters': count_parameters(model),
            'memory_mb': get_model_memory(model),
            'train_time_min': train_time
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Test Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  Test F1 Score: {eval_results['f1_macro']:.2f}%")
        print(f"  Parameters: {results[model_name]['parameters']['total']:,}")
        print(f"  Training Time: {train_time:.1f} minutes")
    
    # Save results
    summary = {name: {k: v for k, v in res.items() if k != 'history'} 
               for name, res in results.items()}
    with open(os.path.join(phase2_dir, 'baseline_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n✅ Phase 2 Complete!")
    
    return results


def run_phase3(data_root: str, output_dir: str) -> dict:
    """Run Phase 3: Transfer Learning."""
    print_banner("PHASE 3: TRANSFER LEARNING", "🧠")
    
    from phase3_transfer_learning import (
        TransferResNet18, TransferEfficientNetB0, TransferDenseNet121,
        create_transfer_data_loaders, DifferentialLearningRate, ModelComparison
    )
    from phase2_baseline_models import Trainer, Evaluator, EarlyStopping, count_parameters, get_model_memory
    
    phase3_dir = os.path.join(output_dir, "phase3_transfer")
    os.makedirs(phase3_dir, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 30
    NUM_CLASSES = 4
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_transfer_data_loaders(
        data_root, batch_size=BATCH_SIZE
    )
    
    comparison = ModelComparison(output_dir=os.path.join(phase3_dir, 'comparison'))
    results = {}
    
    # Models to train
    models = [
        ('ResNet18', TransferResNet18(num_classes=NUM_CLASSES, pretrained=True)),
        ('EfficientNet-B0', TransferEfficientNetB0(num_classes=NUM_CLASSES, pretrained=True)),
        ('DenseNet121', TransferDenseNet121(num_classes=NUM_CLASSES, pretrained=True)),
    ]
    
    for model_name, model in models:
        print(f"\n--- Training {model_name} ---")
        
        start_time = time.time()
        
        # Differential learning rates
        param_groups = DifferentialLearningRate.get_param_groups(
            model, backbone_lr=1e-5, classifier_lr=1e-3
        )
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        early_stopping = EarlyStopping(patience=8, mode='min')
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=os.path.join(phase3_dir, model_name.replace('-', '_'))
        )
        
        history = trainer.train(epochs=EPOCHS, early_stopping=early_stopping)
        train_time = (time.time() - start_time) / 60
        
        trainer.plot_training_curves()
        
        # Evaluate
        best_model = trainer.load_best_model()
        evaluator = Evaluator(best_model, test_loader, class_names, device)
        eval_results = evaluator.evaluate()
        
        cm_path = os.path.join(phase3_dir, model_name.replace('-', '_'), f'{model_name}_confusion_matrix.png')
        evaluator.plot_confusion_matrix(eval_results['confusion_matrix'], cm_path)
        
        result = {
            'history': history,
            'test_accuracy': eval_results['accuracy'],
            'test_f1': eval_results['f1_macro'],
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'parameters': count_parameters(model),
            'memory_mb': get_model_memory(model),
            'train_time_min': train_time
        }
        
        results[model_name] = result
        comparison.add_result(model_name, result)
        
        print(f"\n{model_name} Results:")
        print(f"  Test Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  Test F1 Score: {eval_results['f1_macro']:.2f}%")
        print(f"  Parameters: {result['parameters']['total']:,}")
        print(f"  Training Time: {train_time:.1f} minutes")
    
    # Generate comparisons
    comparison.plot_accuracy_comparison()
    comparison.plot_parameter_comparison()
    comparison.plot_training_dynamics()
    comparison.generate_comparison_table()
    
    # Save results
    summary = {name: {k: v for k, v in res.items() if k != 'history'} 
               for name, res in results.items()}
    with open(os.path.join(phase3_dir, 'transfer_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n✅ Phase 3 Complete!")
    
    return results


def run_phase4(data_root: str, output_dir: str, baseline_results: dict, transfer_results: dict) -> dict:
    """Run Phase 4: Result Analysis."""
    print_banner("PHASE 4: RESULT ANALYSIS", "📊")
    
    from phase4_result_analysis import ResultAnalyzer
    
    phase4_dir = os.path.join(output_dir, "phase4_analysis")
    analyzer = ResultAnalyzer(output_dir=phase4_dir)
    
    # Combine all results
    all_results = {}
    if baseline_results:
        all_results.update(baseline_results)
    if transfer_results:
        all_results.update(transfer_results)
    
    # Generate summary report
    analyzer.generate_summary_report(all_results)
    
    print("\n✅ Phase 4 Complete!")
    
    return {'analyzer': analyzer, 'all_results': all_results}


def create_sample_dataset(uploads_dir: str, sample_dir: str):
    """Create sample dataset from uploaded images."""
    import shutil
    
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)
    
    uploads_path = Path(uploads_dir)
    uploaded_files = sorted(uploads_path.glob("*.png"))
    
    # Create class folders and copy images
    for idx, img_file in enumerate(uploaded_files[:4]):
        class_dir = sample_path / f"class {idx}"
        class_dir.mkdir(exist_ok=True)
        
        # Copy image multiple times to create a minimal dataset
        for i in range(10):
            shutil.copy(img_file, class_dir / f"sample_{idx}_{i}.png")
    
    print(f"Created sample dataset at: {sample_path}")
    return str(sample_path)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Medical Image Classification Pipeline')
    parser.add_argument('--data_root', type=str, default='./sample_data',
                       help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--phases', type=str, default='1,2,3,4',
                       help='Comma-separated list of phases to run (e.g., "1,2,3,4")')
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "🏥" * 35)
    print(" " * 20 + "MEDICAL IMAGE CLASSIFICATION SYSTEM")
    print(" " * 15 + "Automated Chest X-ray Diagnosis using Deep Learning")
    print("🏥" * 35 + "\n")
    
    # Print system info
    print_system_info()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check/create dataset
    data_root = args.data_root
    if not Path(data_root).exists():
        print(f"\nDataset not found at {data_root}")
        print("Creating sample dataset from uploaded images...")
        data_root = create_sample_dataset("/mnt/user-data/uploads", "./sample_data")
    
    # Parse phases to run
    phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
    
    # Store results
    results = {}
    
    # Run phases
    if 1 in phases_to_run:
        results['phase1'] = run_phase1(data_root, str(output_dir))
    
    if 2 in phases_to_run:
        results['phase2'] = run_phase2(data_root, str(output_dir))
    
    if 3 in phases_to_run:
        results['phase3'] = run_phase3(data_root, str(output_dir))
    
    if 4 in phases_to_run:
        baseline_results = results.get('phase2', {})
        transfer_results = results.get('phase3', {})
        results['phase4'] = run_phase4(data_root, str(output_dir), baseline_results, transfer_results)
    
    # Final summary
    print_banner("PIPELINE COMPLETE!", "✨")
    print(f"All results saved to: {output_dir}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── phase1_exploration/  - Data analysis and visualizations")
    print(f"  ├── phase2_baseline/     - Baseline CNN models and results")
    print(f"  ├── phase3_transfer/     - Transfer learning results")
    print(f"  └── phase4_analysis/     - Comprehensive analysis report")
    
    return results


if __name__ == "__main__":
    main()
