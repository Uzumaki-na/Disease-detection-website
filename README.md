# Health Risk Assessment ML Platform

A machine learning platform for medical image classification, focusing on skin cancer and malaria detection using PyTorch and deep learning.

## 🚀 Features
- Skin cancer detection using EfficientNet architecture
- GPU-accelerated model training and inference
- Containerized deployment with Docker
- React frontend with modern UI
- FastAPI backend with async support
- MongoDB for data persistence

## 🛠️ Technical Stack
### Backend
- Python 3.11
- FastAPI
- PyTorch 2.2.1
- TorchVision 0.17.1
- Scikit-learn
- MongoDB (Motor)

### Frontend
- React 18.2.0
- TypeScript
- Tailwind CSS
- Vite

## 🔧 Setup Instructions

### Prerequisites
- Python 3.11+
- Docker Desktop with NVIDIA Container Toolkit
- NVIDIA GPU (optional but recommended)
- Node.js 20+ (for local frontend development)

### Environment Setup
1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create and configure .env file:
```env
MONGODB_URL=mongodb://mongodb:27017
DATABASE_NAME=health_assessment
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
BACKEND_URL=http://localhost:8000
VITE_API_URL=http://localhost:8000/api
MODEL_SAVE_PATH=/app/models
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-api-key
```

3. Train the models:
```bash
cd backend
pip install -r requirements.txt
python train_models_torch.py
```

4. Start the application:
```bash
docker compose up --build
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🏗️ Project Structure
```
.
├── backend/
│   ├── ml/                    # Machine learning models and utilities
│   ├── datasets/              # Dataset management
│   ├── models/               # Trained model files
│   ├── main.py              # FastAPI application
│   ├── train_models_torch.py # Model training script
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/                 # React application source
│   ├── public/              # Static assets
│   └── package.json        # Node.js dependencies
└── docker-compose.yml      # Container orchestration
```

## 🔬 Model Training
The `train_models_torch.py` script:
1. Downloads datasets from Kaggle
2. Preprocesses images and organizes data
3. Trains models using k-fold cross-validation
4. Implements early stopping and learning rate scheduling
5. Saves the best performing models

## 🚀 Development Workflow
1. Backend development:
```bash
cd backend
python -m uvicorn main:app --reload
```

2. Frontend development:
```bash
cd frontend
npm install
npm run dev
```

## 📊 Model Performance
- Skin Cancer Detection:
  * Accuracy: [Your metrics]
  * Precision: [Your metrics]
  * Recall: [Your metrics]

- Malaria Detection:
  * Accuracy: [Your metrics]
  * Precision: [Your metrics]
  * Recall: [Your metrics]

## 🔒 Security Considerations
1. Never commit .env files or API keys
2. Use environment variables for sensitive data
3. Implement rate limiting for API endpoints
4. Validate and sanitize all inputs

## 🤝 Contributing
[Your contribution guidelines]

## 📝 License
[Your license information]
