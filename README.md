# 🏥 Medical Image Classification Platform

An advanced deep learning platform for automated medical image classification, powered by PyTorch and EfficientNet. This system provides state-of-the-art detection capabilities for skin cancer and malaria, helping healthcare professionals make more informed decisions.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.2-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2.0-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 Table of Contents
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔬 Disease Detection
- **Skin Cancer Classification**
  - 7 classes of skin lesions (HAM10000 dataset)
  - High-resolution image processing
  - Detailed probability distribution
  - Lesion localization heatmaps
  
- **Malaria Detection**
  - Binary classification (parasitized/uninfected)
  - Cell image analysis
  - Rapid screening capability
  - Confidence scoring

### 🧠 Machine Learning Pipeline
- **Model Architecture**
  - EfficientNet-B0 backbone
  - Custom classification heads
  - Transfer learning optimization
  - GPU acceleration support
  
- **Training Features**
  - K-fold cross-validation (k=5)
  - Advanced data augmentation
  - Early stopping mechanism
  - Learning rate scheduling
  - Gradient clipping
  - TensorBoard integration

### 💻 Technical Features
- **Backend**
  - FastAPI for high-performance API
  - Async database operations
  - Comprehensive error handling
  - Request validation
  - Rate limiting
  
- **Frontend**
  - React 18 with TypeScript
  - Real-time image processing
  - Interactive visualizations
  - Responsive design
  - Dark/Light theme support

## 🖥️ System Requirements

### Minimum Requirements
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB
- Storage: 10GB free space
- OS: Windows 10/11, Ubuntu 20.04+, or macOS 12+

### Recommended Requirements
- CPU: 8 cores, 3.5GHz+
- RAM: 16GB
- GPU: NVIDIA RTX 3050+ (6GB+ VRAM)
- Storage: 20GB SSD
- OS: Windows 11 or Ubuntu 22.04

### Software Requirements
- Python 3.11+
- Node.js 20+
- MongoDB 6.0+
- NVIDIA CUDA 12.1+ (for GPU support)
- Git 2.3+

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Kaggle API
1. Create a Kaggle account at https://www.kaggle.com
2. Go to 'Account' → 'Create New API Token'
3. Download `kaggle.json` and place it in:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

### 4. Set Up Environment Variables
Create `.env` file in the root directory:
```env
# Security
SECRET_KEY=your-secure-secret-key
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256

# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=health_assessment

# API Configuration
BACKEND_URL=http://localhost:8000
VITE_API_URL=http://localhost:8000/api

# ML Configuration
MODEL_SAVE_PATH=./models
BATCH_SIZE=32
NUM_WORKERS=4
LEARNING_RATE=0.001

# Kaggle Configuration
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

### 5. Install Frontend Dependencies
```bash
cd frontend
npm install
```

## 🎮 Usage

### 1. Start MongoDB
```bash
mongod
```

### 2. Train Models
```bash
cd backend
python train_models_torch.py
```

### 3. Start Backend Server
```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Start Frontend Development Server
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:5173`

## 🧬 Model Architecture

### EfficientNet-B0 Architecture
```
Input (224x224 RGB)
├── EfficientNet-B0 Backbone
│   ├── MBConv blocks
│   ├── Squeeze-and-Excitation
│   └── Feature Pyramid Network
├── Global Average Pooling
├── Dropout (p=0.2)
└── Classification Head
    ├── Skin Cancer: 7 classes
    └── Malaria: 2 classes
```

### Training Configuration
```python
{
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "scheduler": "ReduceLROnPlateau",
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 10,
    "k_folds": 5
}
```

## 📡 API Reference

### Authentication
```http
POST /api/auth/login
POST /api/auth/register
GET /api/auth/me
```

### Predictions
```http
POST /api/predict/skin-cancer
POST /api/predict/malaria
GET /api/predict/history
```

### Model Management
```http
GET /api/models/status
POST /api/models/retrain
GET /api/models/metrics
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/NewFeature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m 'Add NewFeature'`
6. Push: `git push origin feature/NewFeature`
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📊 Performance Metrics

### Skin Cancer Model (HAM10000)
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 91.2%  |
| Precision | 89.8%  |
| Recall    | 90.5%  |
| F1 Score  | 90.1%  |

### Malaria Detection
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 96.5%  |
| Precision | 95.8%  |
| Recall    | 97.2%  |
| F1 Score  | 96.5%  |

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- Rate limiting
- Input sanitization
- CORS protection
- Environment variable security
- File upload validation

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👥 Authors

- **Panav Jain** - [GitHub](https://github.com/panavjain)

## 🙏 Acknowledgments

- [HAM10000 Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [PyTorch Team](https://pytorch.org)

## 📞 Support

For support:
- Open an [issue](https://github.com/yourusername/medical-image-classification/issues)
- Email: your.email@example.com
- Documentation: [Wiki](https://github.com/yourusername/medical-image-classification/wiki)
