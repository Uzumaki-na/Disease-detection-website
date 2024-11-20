from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import logging
import os
from pathlib import Path
from auth import (
    UserCreate,
    User,
    Token,
    create_access_token,
    authenticate_user,
    get_current_user,
    get_password_hash,
    users_db,
    get_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from datetime import timedelta

# Set up logging with more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Risk Assessment API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Model configurations
MODEL_CONFIG = {
    'input_size': (224, 224),
    'batch_size': 1,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Cache for model predictions
from functools import lru_cache
from datetime import datetime

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_health = {}
        self.last_loaded = {}
        
    def load_model(self, model_path, model_name):
        try:
            model = torch.load(model_path, map_location=MODEL_CONFIG['device'])
            model.eval()  # Set to evaluation mode
            self.models[model_name] = model
            self.last_loaded[model_name] = datetime.now()
            self.model_health[model_name] = True
            logger.info(f"Successfully loaded model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}", exc_info=True)
            self.model_health[model_name] = False
            return False

    def get_model(self, model_name):
        return self.models.get(model_name)

    def get_health(self):
        return {
            "models": self.model_health,
            "last_loaded": {k: v.isoformat() for k, v in self.last_loaded.items()},
            "device": str(MODEL_CONFIG['device'])
        }

model_manager = ModelManager()

# Load models at startup
try:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_files = {
        "skin_cancer": models_dir / "skin_cancer_model.pth",
        "malaria": models_dir / "malaria_model.pth"
    }
    
    for model_name, model_path in model_files.items():
        model_manager.load_model(model_path, model_name)
        
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)

# Enhanced image preprocessing with validation
def preprocess_image(image, max_size_mb=10):
    """Preprocess image for model input with size validation"""
    try:
        # Convert image to bytes for size check
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr)
        size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        if size_mb > max_size_mb:
            raise ValueError(f"Image size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)")
            
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(MODEL_CONFIG['device'])
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# Cached prediction function
@lru_cache(maxsize=100)
def get_cached_prediction(image_hash, model_name):
    """Cache for model predictions based on image hash"""
    return None  # Will be populated on first prediction

@app.get("/")
async def root():
    return {"message": "Medical Image Analysis API"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server connectivity"""
    logger.info("Test endpoint called")
    return {"status": "ok", "message": "Server is running"}

@app.post("/register", response_model=User)
async def register(user_data: UserCreate):
    try:
        # Check if user already exists
        existing_user = await get_user(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        user_dict = {
            "email": user_data.email,
            "firstName": user_data.firstName,
            "lastName": user_data.lastName,
            "hashed_password": get_password_hash(user_data.password)
        }
        users_db[user_data.email] = user_dict
        
        logger.info(f"User registered successfully: {user_data.email}")
        
        return {
            "email": user_data.email,
            "firstName": user_data.firstName,
            "lastName": user_data.lastName
        }
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = await authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, expires_delta=access_token_expires
        )
        logger.info(f"Token generated successfully for user: {user['email']}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Error generating token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    try:
        logger.info(f"User retrieved successfully: {current_user.email}")
        return current_user
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": model_manager.get_health()
    }

@app.post("/predict/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):
    """Endpoint for skin cancer prediction with enhanced error handling"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get model
        model = model_manager.get_model("skin_cancer")
        if not model:
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Generate image hash for caching
        image_hash = hash(image_bytes)
        cached_result = get_cached_prediction(image_hash, "skin_cancer")
        if cached_result:
            return cached_result
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.sigmoid(output).cpu().numpy()[0]
        
        result = {
            "prediction": float(probabilities),
            "confidence": float(probabilities) if probabilities >= 0.5 else float(1 - probabilities),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        get_cached_prediction.cache_info()
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in skin cancer prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again."
        )

@app.post("/predict/malaria")
async def predict_malaria(file: UploadFile = File(...)):
    """Endpoint for malaria prediction with enhanced error handling"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get model
        model = model_manager.get_model("malaria")
        if not model:
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Generate image hash for caching
        image_hash = hash(image_bytes)
        cached_result = get_cached_prediction(image_hash, "malaria")
        if cached_result:
            return cached_result
        
        # Make prediction
        with torch.no_grad():
            output = model(processed_image)
            probabilities = torch.sigmoid(output).cpu().numpy()[0]
        
        result = {
            "prediction": float(probabilities),
            "confidence": float(probabilities) if probabilities >= 0.5 else float(1 - probabilities),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        get_cached_prediction.cache_info()
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in malaria prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again."
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")