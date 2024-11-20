import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import os
import logging
import kaggle
from pathlib import Path
import zipfile
import shutil
from PIL import Image
import io
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_gpu():
    """
    Diagnose GPU setup and print detailed information.
    """
    logger.info("=== GPU Diagnostic Information ===")
    
    # Check PyTorch version and CUDA availability
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test GPU with a simple operation
        logger.info("\nTesting GPU with matrix multiplication...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            logger.info("GPU test successful!")
        except Exception as e:
            logger.error(f"GPU test failed: {str(e)}")
    else:
        logger.warning("CUDA is not available. Using CPU only.")

def setup_device():
    """
    Configure device (GPU/CPU) for PyTorch.
    Returns the device to use.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA: {torch.__version__}")
        # Log GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU - No GPU detected!")
        logger.info("Please ensure NVIDIA drivers and CUDA toolkit are installed")
        logger.info(f"PyTorch version: {torch.__version__}")
        if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
            logger.info("CUDA is not available in this PyTorch installation")
    return device

def is_valid_image(image_path):
    """
    Check if the image file is valid and can be opened.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def validate_and_copy_image(src_path, dst_path):
    """
    Validate image before copying to destination.
    """
    if is_valid_image(src_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    return False

def organize_dataset(dataset_name):
    """
    Organize dataset into proper directory structure.
    """
    base_dir = Path("datasets") / dataset_name
    if not base_dir.exists():
        logger.error(f"Dataset directory {base_dir} not found")
        return False
    
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    # Create directories if they don't exist
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    return True

# Custom image loader with error handling
def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {path}: {str(e)}")
        # Create a blank RGB image as fallback
        return Image.new('RGB', (224, 224), 'black')

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, loader=safe_loader)
        self.error_count = 0  # Track consecutive errors
        
    def __getitem__(self, index):
        try:
            self.error_count = 0  # Reset error count on success
            return super().__getitem__(index)
        except Exception as e:
            logger.error(f"Error getting item at index {index}: {str(e)}")
            self.error_count += 1
            if self.error_count > 5:  # Prevent infinite loops
                logger.error("Too many consecutive errors, returning default image")
                img = Image.new('RGB', (224, 224), 'black')
                if self.transform:
                    img = self.transform(img)
                return img, 0  # Return with class 0
            # Try next index
            return self.__getitem__((index + 1) % len(self))

def create_data_loaders(data_dir, batch_size=32, num_workers=4, fold_idx=None, k_folds=5):
    """
    Create PyTorch DataLoaders for training and validation with k-fold cross-validation support.
    """
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Load dataset
        dataset = SafeImageFolder(data_dir, transform=None)  # No transform yet
        logger.info(f"Found {len(dataset)} images in {data_dir}")
        
        if len(dataset) == 0:
            raise ValueError(f"No valid images found in {data_dir}")
        
        # Setup k-fold cross-validation
        if fold_idx is not None:
            # Generate indices for k-fold split
            indices = list(range(len(dataset)))
            np.random.shuffle(indices)
            fold_size = len(dataset) // k_folds
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx != k_folds - 1 else len(dataset)
            
            train_indices = indices[:val_start] + indices[val_end:]
            val_indices = indices[val_start:val_end]
            
            # Create samplers
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            
            # Create datasets with appropriate transforms
            train_dataset = SafeImageFolder(data_dir, transform=train_transform)
            val_dataset = SafeImageFolder(data_dir, transform=val_transform)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            # Regular train/val split
            val_size = int(0.2 * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Apply transforms
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise

class EfficientModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientModel, self).__init__()
        # Load pre-trained EfficientNet using new weights parameter
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze most layers for transfer learning
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Only train the last few layers
        for param in self.model.features[-2:].parameters():
            param.requires_grad = True
            
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Increased dropout
            nn.Linear(1280, 640),  # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(640, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    Train the model using GPU acceleration with improved training pipeline.
    """
    try:
        criterion = nn.CrossEntropyLoss()
        # Separate parameter groups with different learning rates
        optimizer = optim.AdamW([
            {'params': model.model.features.parameters(), 'lr': 1e-5},  # Very small LR for frozen layers
            {'params': model.model.classifier.parameters(), 'lr': 1e-3}  # Larger LR for new layers
        ], weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, min_lr=1e-6)
        
        model = model.to(device)
        scaler = torch.cuda.amp.GradScaler()  # Added automatic mixed precision
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience = 10  # Increased patience
        patience_counter = 0
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 
            'val_acc': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 10 == 0:
                        logger.info(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}')
                        
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            if batch_count == 0:
                logger.error("No successful training batches in this epoch")
                continue
                
            train_loss = train_loss / batch_count
            history['train_loss'].append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        val_batch_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue
            
            if val_batch_count == 0:
                logger.error("No successful validation batches in this epoch")
                continue
            
            # Calculate metrics
            val_loss = val_loss / val_batch_count
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            accuracy = (all_preds == all_labels).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            conf_matrix = confusion_matrix(all_labels, all_preds)
            
            # Update history
            history['val_loss'].append(val_loss)
            history['val_acc'].append(accuracy)
            history['val_precision'].append(precision)
            history['val_recall'].append(recall)
            history['val_f1'].append(f1)
            
            # Logging
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}')
            logger.info(f'Val Accuracy: {accuracy*100:.2f}%')
            logger.info(f'Val Precision: {precision:.4f}')
            logger.info(f'Val Recall: {recall:.4f}')
            logger.info(f'Val F1: {f1:.4f}')
            logger.info(f'Confusion Matrix:\n{conf_matrix}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model (both by loss and accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                }, 'best_model_by_loss.pth')
                logger.info(f'Saved new best model (by loss) with validation loss: {val_loss:.4f}')
            
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': accuracy,
                    'val_metrics': {
                        'loss': val_loss,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                }, 'best_model_by_accuracy.pth')
                logger.info(f'Saved new best model (by accuracy) with validation accuracy: {accuracy*100:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def download_and_extract_dataset(dataset_name):
    """
    Download and extract dataset from Kaggle with proper train/val split.
    """
    logger.info(f"Downloading {dataset_name} dataset...")
    base_dir = Path("datasets") / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check available disk space (in bytes)
        import shutil
        total, used, free = shutil.disk_usage("/")
        required_space = {
            "malaria": 500 * 1024 * 1024,  # 500MB
            "skin_cancer": 3 * 1024 * 1024 * 1024  # 3GB
        }

        if free < required_space.get(dataset_name, 5 * 1024 * 1024 * 1024):
            raise OSError(f"Not enough disk space. Need at least {required_space.get(dataset_name) / (1024*1024*1024):.1f}GB free")

        if dataset_name == "malaria":
            # Download malaria dataset
            kaggle.api.dataset_download_files(
                'iarunava/cell-images-for-detecting-malaria',
                path=str(base_dir),
                unzip=True
            )
            
            # Organize malaria dataset
            cell_images = base_dir / "cell_images"
            if cell_images.exists():
                # Create train and validation directories
                for split in ["train", "val"]:
                    for category in ["Parasitized", "Uninfected"]:
                        split_dir = base_dir / split / category
                        split_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each category
                for category in ["Parasitized", "Uninfected"]:
                    src_dir = cell_images / category
                    if src_dir.exists():
                        # Get all valid images
                        valid_images = [img for img in src_dir.glob("*.png") if is_valid_image(str(img))]
                        
                        # Shuffle images
                        import random
                        random.shuffle(valid_images)
                        
                        # Split into train (80%) and validation (20%)
                        split_idx = int(len(valid_images) * 0.8)
                        train_images = valid_images[:split_idx]
                        val_images = valid_images[split_idx:]
                        
                        # Copy train images
                        for img in train_images:
                            try:
                                shutil.copy2(str(img), str(base_dir / "train" / category / img.name))
                            except Exception as e:
                                logger.warning(f"Failed to copy train image {img}: {e}")
                                
                        # Copy validation images
                        for img in val_images:
                            try:
                                shutil.copy2(str(img), str(base_dir / "val" / category / img.name))
                            except Exception as e:
                                logger.warning(f"Failed to copy validation image {img}: {e}")
                
                # Clean up only after successful copy
                try:
                    shutil.rmtree(str(cell_images))
                except Exception as e:
                    logger.warning(f"Failed to clean up {cell_images}: {e}")
            
        elif dataset_name == "skin_cancer":
            # Create temporary download directory
            temp_dir = base_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Download skin cancer dataset to temp directory
                kaggle.api.dataset_download_files(
                    'kmader/skin-cancer-mnist-ham10000',
                    path=str(temp_dir),
                    unzip=True
                )
                
                # Process files in smaller batches
                ham_images = temp_dir / "HAM10000_images_part_1"
                metadata = temp_dir / "HAM10000_metadata.csv"
                
                if ham_images.exists() and metadata.exists():
                    import pandas as pd
                    df = pd.read_csv(metadata)
                    
                    # Create directories for each class
                    classes = df['dx'].unique()
                    for cls in classes:
                        (base_dir / "train" / cls).mkdir(parents=True, exist_ok=True)
                        (base_dir / "val" / cls).mkdir(parents=True, exist_ok=True)
                    
                    # Process images in batches
                    batch_size = 100
                    for i in range(0, len(df), batch_size):
                        batch_df = df.iloc[i:i+batch_size]
                        
                        for _, row in batch_df.iterrows():
                            img_id = row['image_id']
                            dx = row['dx']
                            src_path = ham_images / f"{img_id}.jpg"
                            if src_path.exists():
                                try:
                                    if is_valid_image(str(src_path)):
                                        if random.random() < 0.8:
                                            dst_path = base_dir / "train" / dx / f"{img_id}.jpg"
                                        else:
                                            dst_path = base_dir / "val" / dx / f"{img_id}.jpg"
                                        shutil.copy2(str(src_path), str(dst_path))
                                except Exception as e:
                                    logger.warning(f"Failed to process {img_id}: {e}")
                
                # Clean up temp directory
                try:
                    shutil.rmtree(str(temp_dir))
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

            except Exception as e:
                logger.error(f"Error downloading/extracting skin cancer dataset: {e}")
                # Clean up temp directory on failure
                if temp_dir.exists():
                    try:
                        shutil.rmtree(str(temp_dir))
                    except:
                        pass
                raise

        logger.info(f"Successfully downloaded and organized {dataset_name} dataset")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {str(e)}")
        return False

def check_and_download_datasets():
    """
    Check if datasets exist and download if needed.
    """
    datasets = ["malaria", "skin_cancer"]
    for dataset in datasets:
        base_dir = Path("datasets") / dataset
        train_dir = base_dir / "train"
        val_dir = base_dir / "val"
        
        # Download if dataset doesn't exist or is empty
        if not train_dir.exists() or not any(train_dir.iterdir()) or not val_dir.exists() or not any(val_dir.iterdir()):
            logger.info(f"{dataset} dataset not found or empty. Downloading...")
            if not download_and_extract_dataset(dataset):
                logger.error(f"Failed to download {dataset} dataset")
                continue
            
            # Verify dataset structure
            if not organize_dataset(dataset):
                logger.error(f"Failed to organize {dataset} dataset")
                continue
        else:
            logger.info(f"{dataset} dataset already exists")

def train_models():
    """
    Main function to train models with k-fold cross-validation.
    """
    device = setup_device()
    diagnose_gpu()
    
    # Check and download datasets if needed
    check_and_download_datasets()
    
    k_folds = 5
    
    # Train skin cancer model
    logger.info("Training skin cancer detection model...")
    skin_data_dir = "datasets/skin_cancer"
    if os.path.exists(skin_data_dir):
        # K-fold cross-validation
        fold_histories = []
        for fold in range(k_folds):
            logger.info(f"Training fold {fold+1}/{k_folds}")
            train_loader, val_loader = create_data_loaders(skin_data_dir, fold_idx=fold, k_folds=k_folds)
            model = EfficientModel(num_classes=7)
            _, history = train_model(model, train_loader, val_loader, device)
            fold_histories.append(history)
        
        # Calculate and log average metrics across folds
        avg_metrics = {
            'val_acc': np.mean([h['val_acc'][-1] for h in fold_histories]),
            'val_precision': np.mean([h['val_precision'][-1] for h in fold_histories]),
            'val_recall': np.mean([h['val_recall'][-1] for h in fold_histories]),
            'val_f1': np.mean([h['val_f1'][-1] for h in fold_histories])
        }
        logger.info("Average metrics across folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Train malaria model
    logger.info("Training malaria detection model...")
    malaria_data_dir = "datasets/malaria"
    if os.path.exists(malaria_data_dir):
        # K-fold cross-validation
        fold_histories = []
        for fold in range(k_folds):
            logger.info(f"Training fold {fold+1}/{k_folds}")
            train_loader, val_loader = create_data_loaders(malaria_data_dir, fold_idx=fold, k_folds=k_folds)
            model = EfficientModel(num_classes=2)
            _, history = train_model(model, train_loader, val_loader, device)
            fold_histories.append(history)
        
        # Calculate and log average metrics across folds
        avg_metrics = {
            'val_acc': np.mean([h['val_acc'][-1] for h in fold_histories]),
            'val_precision': np.mean([h['val_precision'][-1] for h in fold_histories]),
            'val_recall': np.mean([h['val_recall'][-1] for h in fold_histories]),
            'val_f1': np.mean([h['val_f1'][-1] for h in fold_histories])
        }
        logger.info("Average metrics across folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    train_models()
