# EMNIST CNN Classifier 🤖📝

## ___DISCLAIMER___
* *This project is done only for __educational purposes__*
* *If you are interested in learning deep learning and CNN architectures, you can learn through my implementation*
* *I am __not responsible__ for any misuse of this code or models*
* *This project is intended for __academic and portfolio demonstration__ purposes only*

A comprehensive deep learning project implementing **Convolutional Neural Networks (CNNs)** for handwritten character recognition using the **Extended MNIST (EMNIST)** dataset. This project demonstrates advanced machine learning techniques, regularization methods, and model optimization strategies.

## 🎯 Project Overview

This project develops and compares multiple CNN architectures with different regularization techniques to classify handwritten alphanumeric characters (A-Z, 0-9) from the EMNIST dataset. The implementation showcases **state-of-the-art deep learning practices** and **systematic model evaluation**.

### 🔑 Key Achievements
- **91.45%** best accuracy achieved with Batch Normalization
- **36-class classification** (26 letters + 10 digits)
- **100,800 images** processed and analyzed
- **5 different regularization techniques** implemented and compared
- Comprehensive model evaluation with detailed performance metrics

## 📊 Dataset Information

- **Total Images**: 100,800 samples
- **Image Dimensions**: 28×28 pixels (grayscale)
- **Classes**: 36 (A-Z letters + 0-9 digits)
- **Distribution**: Balanced dataset (~2,800 images per class)
- **Format**: Preprocessed PyTorch tensors with normalization

## 🏗️ Architecture & Implementation

### Core CNN Architecture
```python
- 5 Convolutional Layers (1→32→64→128→256→128 channels)
- Mixed Pooling Strategy (MaxPool2d + AvgPool2d)
- Multiple Activation Functions (ReLU, GELU, Tanh)
- 2 Fully Connected Layers (256→36 outputs)
- Dropout Regularization (50%)
- Total Parameters: ~725k
```

### 🛠️ Technical Stack
- **Framework**: PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Scikit-learn
- **Device**: MPS (Apple Silicon) / CUDA / CPU
- **Data Processing**: torchvision transforms

## 🎛️ Regularization Techniques Implemented

| Method | Accuracy | Loss | Training Time | Description |
|--------|----------|------|---------------|-------------|
| **Batch Normalization** | **91.45%** | **0.241** | 127.2s | ✅ **Best Overall** |
| L2 Regularization | 91.10% | 0.253 | 352.2s | Weight decay optimization |
| Learning Rate Scheduler | 91.09% | 0.258 | 114.1s | Adaptive learning rate |
| Base Model | 90.94% | 0.263 | 185.2s | Baseline implementation |
| Early Stopping | 90.17% | 0.276 | 72.8s | Fastest training |

### 🏆 Best Model: Batch Normalization CNN
- **Architecture**: Enhanced with BatchNorm2d layers
- **Performance**: 91.45% test accuracy
- **Features**: Improved gradient flow and training stability
- **Optimization**: Adam optimizer with 0.001 learning rate

## 📈 Model Performance Analysis

### Training Metrics
- **Epochs**: 10 (with early stopping when applicable)
- **Batch Size**: 64
- **Train/Validation/Test Split**: 64%/16%/20%
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with various configurations

### Evaluation Metrics
- ✅ **Accuracy**: Up to 91.45%
- ✅ **Precision, Recall, F1-Score**: Macro-averaged
- ✅ **Confusion Matrix**: 36×36 class visualization
- ✅ **ROC Curves**: Multi-class analysis
- ✅ **Training/Validation Loss Curves**

## 🔍 Key Technical Features

### 1. **Advanced Data Pipeline**
```python
- Custom ImageFolder dataset loading
- Comprehensive data augmentation
- Balanced train/validation/test splits
- Efficient DataLoader implementation
```

### 2. **Model Architecture Innovation**
- **Mixed Pooling Strategy**: Combines MaxPool and AvgPool
- **Multi-Activation Design**: ReLU, GELU, and Tanh functions
- **Progressive Feature Maps**: 1→32→64→128→256→128 channels
- **Flexible Dropout**: Configurable regularization

### 3. **Comprehensive Evaluation**
- **Multi-class ROC Analysis**: 36 individual class curves
- **Detailed Classification Reports**: Per-class metrics
- **Visual Performance Tracking**: Loss and accuracy plots
- **Confusion Matrix Heatmaps**: Detailed error analysis

### 4. **Regularization Experiments**
- **Early Stopping**: Patience-based training termination
- **Batch Normalization**: Internal covariate shift reduction
- **Learning Rate Scheduling**: Linear decay implementation
- **L2 Regularization**: Weight decay prevention
- **Dropout**: Stochastic regularization

## 📁 Project Structure

```
emnist-cnn-classifier/
├── cnn-emnist.ipynb           # Main implementation notebook
├── cnn_dataset/               # EMNIST dataset organized by class
│   ├── 0/                     # Digit '0' images
│   ├── 1/                     # Digit '1' images
│   └── ...                    # Other classes (A-Z, 2-9)
├── weights/                   # Saved model weights
│   └── part3_weight.pt        # Best performing model
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy torchinfo
```

### Running the Project
1. **Clone the repository**
2. **Install dependencies** (see prerequisites)
3. **Open `cnn-emnist.ipynb`** in Jupyter
4. **Run all cells** to reproduce results
5. **Experiment** with different configurations

### Model Loading
```python
# Load the best performing model
model = BatchNormCNN(dropout_perc=0.5)
model.load_state_dict(torch.load('weights/part3_weight.pt'))
model.eval()
```

## 📊 Visualizations & Analysis

The project includes comprehensive visualizations:
- **Dataset exploration** with sample images by class
- **Training progress** with loss/accuracy curves
- **Performance comparison** across all models
- **Confusion matrices** for error analysis
- **ROC curves** for multi-class evaluation
- **Pixel intensity distributions**

## Future Improvements

- **Data Augmentation**: Rotation, scaling, noise injection
- **Advanced Architectures**: ResNet, DenseNet implementations
- **Ensemble Methods**: Model combination strategies
- **Transfer Learning**: Pre-trained model fine-tuning
- **Deployment**: Model serving and API development