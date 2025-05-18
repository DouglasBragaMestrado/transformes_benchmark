# Leukemic Cell Classification Benchmark

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![Transformers](https://img.shields.io/badge/Transformers-latest-green.svg)

## 📝 Overview

This repository contains the implementation of the experiments detailed in the research paper "Comparative Analysis of ConvNeXt, Vision Transformer, and Swin Transformer for Leukemic Cell Classification." The study benchmarks three state-of-the-art deep learning architectures to classify leukemic versus healthy cells from microscopic images.

## 🔍 Research Context

Acute lymphoblastic leukemia (ALL) is the most prevalent childhood cancer, accounting for approximately 25% of pediatric cancer cases. Early and accurate detection is crucial for improving treatment outcomes. This research evaluates and compares the performance of three cutting-edge architectures:

- **ConvNeXt-Base**: A modernized CNN architecture incorporating design principles from transformers
- **Vision Transformer (ViT-Base)**: A pure transformer approach that processes image patches without convolutions
- **Swin Transformer (Swin-Base)**: A hierarchical transformer with shifted window attention mechanism

The experiments utilize the ISBI 2019 Challenge dataset, implementing a rigorous patient-level data split to prevent data leakage and ensure reliable performance metrics.

## 🧠 Key Findings

- Vision Transformer (ViT-Base) achieved the highest test set F1-score of 0.9536
- ConvNeXt-Base demonstrated the fastest convergence and best computational efficiency
- Swin Transformer showed competitive performance but required more training time
- The hybrid ensemble combining CNN and transformer approaches improved generalization for edge cases

## 🛠️ Technologies

The implementation uses the following key technologies:

- **PyTorch**: Deep learning framework for model implementation
- **Hugging Face Transformers**: Pre-trained model library
- **NumPy/Pandas**: Data processing and analysis
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Evaluation metrics

## 📊 Project Structure

```
├── dataset/
│   ├── processado/         # Processed and split dataset
│   ├── train/              # Training images
│   ├── validation/         # Validation images
│   └── test/               # Test images
├── models/                 # Saved model checkpoints
│   └── benchmark_facebook/ # Model weights
├── results/                # Experiment results
│   └── plots/              # Performance visualizations
└── app.ipynb               # Main experiment notebook
```

## 📋 Implementation Details

The implementation includes:

- Patient-level data splitting for rigorous evaluation
- Comprehensive data augmentation pipeline
- Standardized training protocol across architectures
- Early stopping and learning rate scheduling
- Hybrid ensemble model combining CNN and transformer approaches
- Extensive visualization and analysis of model performance

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
python run_benchmark.py
```

Or open and run the Jupyter notebook:

```bash
jupyter notebook app.ipynb
```

## 📈 Results Visualization

The repository includes comprehensive visualization tools for model performance analysis:

- Learning curves
- Confusion matrices
- ROC curves
- Feature attention maps
- Comparative performance radar charts

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{braga2024comparative,
  title={Comparative Analysis of ConvNeXt, Vision Transformer, and Swin Transformer for Leukemic Cell Classification},
  author={Braga, Douglas Costa and Dantas, Daniel Oliveira},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

## 🤝 Acknowledgments

We thank the organizers of the ISBI 2019 Challenge for providing the dataset used in this study.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
