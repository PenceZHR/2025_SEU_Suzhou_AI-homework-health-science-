"""
Phase 1: Data Exploration for Medical Image Classification
============================================================
This module handles data loading, exploration, and preprocessing for
the medical image classification task (chest X-ray diagnosis).

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')


class MedicalDataExplorer:
    """
    A comprehensive data exploration class for medical imaging datasets.
    
    This class provides functionality for:
    - Loading and analyzing image datasets
    - Computing statistical characteristics
    - Visualizing data distributions
    - Preprocessing images for deep learning
    """
    
    # Class labels mapping (customize based on actual dataset)
    CLASS_NAMES = {
        0: "Class 0 (Normal/Healthy)",
        1: "Class 1 (Pathology Type 1)",
        2: "Class 2 (Pathology Type 2)", 
        3: "Class 3 (Pathology Type 3)"
    }
    
    def __init__(self, data_root: str, output_dir: str = "./exploration_results"):
        """
        Initialize the Medical Data Explorer.
        
        Args:
            data_root: Root directory containing class folders
            output_dir: Directory to save exploration results
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_dirs = {}
        self.image_paths = {}
        self.statistics = {}
        
    def discover_dataset(self) -> Dict[str, int]:
        """
        Discover and catalog the dataset structure.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        print("=" * 60)
        print("PHASE 1: DATA EXPLORATION")
        print("=" * 60)
        print(f"\nScanning dataset at: {self.data_root}")
        
        class_counts = {}
        
        # Look for class folders (class 0, class 1, etc.)
        for class_idx in range(10):  # Check up to 10 classes
            for pattern in [f"class {class_idx}", f"class_{class_idx}", f"Class {class_idx}", str(class_idx)]:
                class_dir = self.data_root / pattern
                if class_dir.exists() and class_dir.is_dir():
                    self.class_dirs[class_idx] = class_dir
                    
                    # Count images
                    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                             list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
                    self.image_paths[class_idx] = images
                    class_counts[class_idx] = len(images)
                    break
        
        print(f"\nDiscovered {len(class_counts)} classes:")
        print("-" * 40)
        total = 0
        for class_idx, count in sorted(class_counts.items()):
            class_name = self.CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            print(f"  {class_name}: {count:,} images")
            total += count
        print("-" * 40)
        print(f"  TOTAL: {total:,} images")
        
        return class_counts
    
    def analyze_image_properties(self, sample_size: int = 100) -> Dict:
        """
        Analyze image properties (dimensions, channels, intensity).
        
        Args:
            sample_size: Number of images to sample per class
            
        Returns:
            Dictionary containing image statistics
        """
        print("\n" + "=" * 60)
        print("ANALYZING IMAGE PROPERTIES")
        print("=" * 60)
        
        all_widths = []
        all_heights = []
        all_channels = []
        intensity_stats = {class_idx: [] for class_idx in self.image_paths.keys()}
        
        for class_idx, paths in self.image_paths.items():
            # Sample images
            sample_paths = paths[:min(sample_size, len(paths))]
            
            for img_path in sample_paths:
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    all_widths.append(w)
                    all_heights.append(h)
                    
                    # Determine channels
                    if img.mode == 'L':
                        all_channels.append(1)
                    elif img.mode == 'RGB':
                        all_channels.append(3)
                    elif img.mode == 'RGBA':
                        all_channels.append(4)
                    else:
                        all_channels.append(len(img.getbands()))
                    
                    # Compute intensity statistics (convert to grayscale if needed)
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray)
                    intensity_stats[class_idx].append({
                        'mean': np.mean(img_array),
                        'std': np.std(img_array),
                        'min': np.min(img_array),
                        'max': np.max(img_array)
                    })
                    
                except Exception as e:
                    print(f"  Warning: Could not process {img_path}: {e}")
        
        # Compile statistics
        self.statistics = {
            'dimensions': {
                'width': {'min': min(all_widths), 'max': max(all_widths), 
                         'mean': np.mean(all_widths), 'std': np.std(all_widths)},
                'height': {'min': min(all_heights), 'max': max(all_heights),
                          'mean': np.mean(all_heights), 'std': np.std(all_heights)},
                'unique_sizes': len(set(zip(all_widths, all_heights)))
            },
            'channels': {
                'distribution': dict(Counter(all_channels)),
                'most_common': Counter(all_channels).most_common(1)[0][0]
            },
            'intensity_by_class': {}
        }
        
        # Aggregate intensity stats by class
        for class_idx, stats_list in intensity_stats.items():
            if stats_list:
                self.statistics['intensity_by_class'][class_idx] = {
                    'mean': np.mean([s['mean'] for s in stats_list]),
                    'std': np.mean([s['std'] for s in stats_list]),
                    'min': np.min([s['min'] for s in stats_list]),
                    'max': np.max([s['max'] for s in stats_list])
                }
        
        # Print results
        print("\n📐 IMAGE DIMENSIONS:")
        print(f"  Width  - Min: {self.statistics['dimensions']['width']['min']}, "
              f"Max: {self.statistics['dimensions']['width']['max']}, "
              f"Mean: {self.statistics['dimensions']['width']['mean']:.1f}")
        print(f"  Height - Min: {self.statistics['dimensions']['height']['min']}, "
              f"Max: {self.statistics['dimensions']['height']['max']}, "
              f"Mean: {self.statistics['dimensions']['height']['mean']:.1f}")
        print(f"  Unique sizes: {self.statistics['dimensions']['unique_sizes']}")
        
        print("\n🎨 CHANNELS:")
        for ch, count in self.statistics['channels']['distribution'].items():
            print(f"  {ch}-channel images: {count}")
        
        print("\n💡 INTENSITY STATISTICS BY CLASS:")
        for class_idx, stats in self.statistics['intensity_by_class'].items():
            class_name = self.CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            print(f"  {class_name}:")
            print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        
        return self.statistics
    
    def visualize_sample_images(self, samples_per_class: int = 4) -> str:
        """
        Create a grid visualization of sample images from each class.
        
        Args:
            samples_per_class: Number of samples to show per class
            
        Returns:
            Path to saved visualization
        """
        print("\n" + "=" * 60)
        print("GENERATING SAMPLE IMAGE VISUALIZATION")
        print("=" * 60)
        
        num_classes = len(self.image_paths)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                  figsize=(3*samples_per_class, 3*num_classes))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for row, (class_idx, paths) in enumerate(sorted(self.image_paths.items())):
            class_name = self.CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            
            # Select random samples
            sample_paths = np.random.choice(paths, 
                                            min(samples_per_class, len(paths)), 
                                            replace=False)
            
            for col, img_path in enumerate(sample_paths):
                ax = axes[row, col]
                try:
                    img = Image.open(img_path).convert('L')
                    ax.imshow(img, cmap='gray')
                    if col == 0:
                        ax.set_ylabel(class_name, fontsize=10, fontweight='bold')
                except:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'sample_images.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved sample visualization to: {save_path}")
        return str(save_path)
    
    def visualize_class_distribution(self) -> str:
        """
        Create a bar chart showing the class distribution.
        
        Returns:
            Path to saved visualization
        """
        print("\nGenerating class distribution visualization...")
        
        classes = []
        counts = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.image_paths)))
        
        for class_idx in sorted(self.image_paths.keys()):
            class_name = self.CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            classes.append(f"Class {class_idx}")
            counts.append(len(self.image_paths[class_idx]))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add total annotation
        total = sum(counts)
        ax.text(0.98, 0.98, f'Total: {total:,} images', transform=ax.transAxes,
               ha='right', va='top', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'class_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved class distribution to: {save_path}")
        return str(save_path)
    
    def visualize_intensity_distributions(self, sample_size: int = 200) -> str:
        """
        Create intensity histogram for each class.
        
        Args:
            sample_size: Number of images to sample per class
            
        Returns:
            Path to saved visualization
        """
        print("\nGenerating intensity distribution visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for idx, (class_idx, paths) in enumerate(sorted(self.image_paths.items())):
            if idx >= 4:
                break
                
            ax = axes[idx]
            class_name = self.CLASS_NAMES.get(class_idx, f"Class {class_idx}")
            
            # Sample images and compute histogram
            sample_paths = np.random.choice(paths, min(sample_size, len(paths)), replace=False)
            all_intensities = []
            
            for img_path in sample_paths:
                try:
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img).flatten()
                    # Subsample to keep memory manageable
                    all_intensities.extend(img_array[::10])
                except:
                    pass
            
            ax.hist(all_intensities, bins=50, color=colors[idx], alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Pixel Intensity (0-255)')
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(all_intensities), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(all_intensities):.1f}')
            ax.legend()
        
        plt.suptitle('Pixel Intensity Distributions by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'intensity_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved intensity distributions to: {save_path}")
        return str(save_path)
    
    def generate_exploration_report(self) -> str:
        """
        Generate a comprehensive JSON report of the data exploration.
        
        Returns:
            Path to saved report
        """
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
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
        
        report = {
            'dataset_summary': {
                'total_classes': len(self.image_paths),
                'total_images': sum(len(paths) for paths in self.image_paths.values()),
                'images_per_class': {str(k): len(v) for k, v in self.image_paths.items()}
            },
            'image_statistics': convert_to_serializable(self.statistics),
            'class_names': {str(k): v for k, v in self.CLASS_NAMES.items() 
                           if k in self.image_paths}
        }
        
        save_path = self.output_dir / 'exploration_report.json'
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Saved exploration report to: {save_path}")
        return str(save_path)


def create_sample_dataset_structure(base_dir: str = "./sample_data"):
    """
    Create a sample dataset structure for demonstration.
    This can be used when the actual dataset is not available.
    """
    import shutil
    
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Copy uploaded images as samples
    uploads_dir = Path("/mnt/user-data/uploads")
    
    uploaded_files = sorted(uploads_dir.glob("*.png"))
    
    for idx, img_file in enumerate(uploaded_files[:4]):
        class_dir = base_path / f"class {idx}"
        class_dir.mkdir(exist_ok=True)
        shutil.copy(img_file, class_dir / f"sample_{idx}.png")
    
    print(f"Created sample dataset at: {base_path}")
    return str(base_path)


if __name__ == "__main__":
    # Main execution
    print("\n" + "🏥" * 30)
    print("MEDICAL IMAGE CLASSIFICATION - DATA EXPLORATION")
    print("🏥" * 30 + "\n")
    
    # Try to find dataset or create sample
    data_root = None
    
    # Common paths to check
    possible_paths = [
        "./Dataset2",
        "../Dataset2", 
        "/mnt/user-data/uploads/Dataset2",
        "./data/Dataset2"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            data_root = path
            break
    
    if data_root is None:
        print("Dataset2 not found. Creating sample dataset from uploaded images...")
        data_root = create_sample_dataset_structure()
    
    # Run exploration
    explorer = MedicalDataExplorer(
        data_root=data_root,
        output_dir="./exploration_results"
    )
    
    # Execute exploration pipeline
    class_counts = explorer.discover_dataset()
    
    if class_counts:
        statistics = explorer.analyze_image_properties(sample_size=100)
        explorer.visualize_sample_images(samples_per_class=4)
        explorer.visualize_class_distribution()
        explorer.visualize_intensity_distributions(sample_size=200)
        explorer.generate_exploration_report()
        
        print("\n" + "=" * 60)
        print("✅ DATA EXPLORATION COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved to: {explorer.output_dir}")
    else:
        print("\n⚠️  No images found. Please check dataset path.")
