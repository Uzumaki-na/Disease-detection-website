from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import tensorflow as tf
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
from datetime import timedelta, datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Risk Assessment API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
try:
    logger.info("Loading models...")
    models_dir = Path("models")
    
    # Load skin cancer model
    skin_cancer_model = tf.keras.models.load_model(str(models_dir / "skin_cancer_model.h5"))
    logger.info("Skin cancer model loaded successfully")
    
    # Load malaria model
    malaria_model = tf.keras.models.load_model(str(models_dir / "malaria_model.h5"))
    logger.info("Malaria model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    return img_array

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

@app.post("/predict/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):
    try:
        logger.info("Starting skin cancer prediction")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = skin_cancer_model.predict(processed_image)
        probability = float(prediction[0][0])
        
        result = {
            "prediction": "Malignant" if probability >= 0.5 else "Benign",
            "probability": probability,
            "confidence": probability if probability >= 0.5 else (1 - probability)
        }
        
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/malaria")
async def predict_malaria(file: UploadFile = File(...)):
    try:
        logger.info("Starting malaria prediction")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = malaria_model.predict(processed_image)
        probability = float(prediction[0][0])
        
        result = {
            "prediction": "Parasitized" if probability >= 0.5 else "Uninfected",
            "probability": probability,
            "confidence": probability if probability >= 0.5 else (1 - probability)
        }
        
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "skin_cancer": skin_cancer_model is not None,
            "malaria": malaria_model is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")