import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, models, datasets
import numpy as np
from pathlib import Path
import logging
import psutil
import GPUtil
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_gpu_utilization():
    """Print GPU memory usage"""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming using first GPU
        logger.info(f'GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)')
        logger.info(f'GPU Load: {gpu.load*100:.1f}%')

def print_memory_usage():
    """Print system memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f'RAM Memory: {memory_info.rss / 1024/1024:.1f}MB')
    logger.info(f'Virtual Memory: {memory_info.vms / 1024/1024:.1f}MB')

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
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Set GPU to maximum performance mode
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
        return device
    return torch.device("cpu")

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
            return super().__getitem__(index)
        except Exception as e:
            logger.warning(f"Error loading image at index {index}: {str(e)}")
            self.error_count += 1
            if self.error_count > 10:  # Avoid infinite loops
                raise Exception("Too many consecutive errors loading images")
            # Try next index
            return self.__getitem__((index + 1) % len(self))

def create_data_loaders(data_dir, batch_size=256, num_workers=4, fold_idx=None, k_folds=5):
    """Fast data loading with minimal augmentation for speed"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Keep antialiasing for stability
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if fold_idx is not None:
        dataset = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=None)
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        fold_size = len(dataset) // k_folds
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx != k_folds - 1 else len(dataset)
        
        train_indices = indices[:val_start] + indices[val_end:]
        val_indices = indices[val_start:val_end]
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_dataset = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=transform)
        val_dataset = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    else:
        train_dataset = datasets.ImageFolder(str(Path(data_dir) / "train"), transform=transform)
        val_dataset = datasets.ImageFolder(str(Path(data_dir) / "val"), transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    return train_loader, val_loader

class EfficientModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # More stable learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=50.0
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter('runs/training_metrics')
    
    # Enable automatic mixed precision with more stable settings
    scaler = torch.amp.GradScaler()  # Updated to new API
    
    best_acc = 0.0
    best_model = None
    best_metrics = None
    
    logger.info("=== Starting Training ===")
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use mixed precision training with updated API
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            epoch_loss += loss.item()
            with torch.no_grad():
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Store predictions for epoch metrics
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            # Print batch progress with running metrics
            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                accuracy = 100. * correct / total
                logger.info(
                    f'Epoch: {epoch + 1} | '
                    f'Batch: {batch_idx + 1}/{len(train_loader)} | '
                    f'Loss: {avg_loss:.3f} | '
                    f'Running Acc: {accuracy:.2f}% | '
                    f'LR: {scheduler.get_last_lr()[0]:.6f}'
                )
                running_loss = 0.0
        
        # Calculate epoch training metrics
        train_acc = 100. * correct / total
        epoch_loss = epoch_loss / len(train_loader)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_train_labels, all_train_preds, average='weighted'
        )
        
        # Log training metrics to tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, average='weighted'
        )
        
        # Log validation metrics to tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        # Print epoch summary
        logger.info(
            f'\nEpoch {epoch + 1} Summary:\n'
            f'Training:\n'
            f'  Loss: {epoch_loss:.3f}\n'
            f'  Accuracy: {train_acc:.2f}%\n'
            f'  Precision: {train_precision:.3f}\n'
            f'  Recall: {train_recall:.3f}\n'
            f'  F1: {train_f1:.3f}\n'
            f'Validation:\n'
            f'  Loss: {val_loss:.3f}\n'
            f'  Accuracy: {val_acc:.2f}%\n'
            f'  Precision: {val_precision:.3f}\n'
            f'  Recall: {val_recall:.3f}\n'
            f'  F1: {val_f1:.3f}'
        )
        
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
            best_metrics = {
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            }
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'metrics': best_metrics
            }, 'best_model.pth')
            logger.info(f'\nSaved new best model with validation accuracy: {val_acc:.2f}%')
        
        # Clear GPU cache after each epoch
        torch.cuda.empty_cache()
    
    writer.close()
    
    # Print best model summary
    if best_metrics:
        logger.info(
            f'\nBest Model Performance (Epoch {best_metrics["epoch"]}):\n'
            f'Validation Accuracy: {best_metrics["val_acc"]:.2f}%\n'
            f'Validation Loss: {best_metrics["val_loss"]:.3f}\n'
            f'Validation F1: {best_metrics["val_f1"]:.3f}\n'
            f'Validation Precision: {best_metrics["val_precision"]:.3f}\n'
            f'Validation Recall: {best_metrics["val_recall"]:.3f}'
        )
    
    return model, best_model

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
    try:
        # Initialize GPU
        device = setup_device()
        diagnose_gpu()  # Print GPU info
        logger.info(f"Using device: {device}")
        
        # Train skin cancer detection model
        logger.info("Training skin cancer detection model...")
        skin_data_dir = Path("backend/datasets/skin_cancer")
        
        if not skin_data_dir.exists():
            logger.error(f"Skin cancer dataset directory not found at {skin_data_dir}")
            return
            
        if not (skin_data_dir / "train").exists() or not any((skin_data_dir / "train").iterdir()):
            logger.error("Skin cancer dataset not found or empty. Skipping training.")
            return
            
        # Use k-fold cross-validation
        k_folds = 5
        
        for fold in range(k_folds):
            logger.info(f"Training fold {fold + 1}/{k_folds}")
            
            try:
                # Create data loaders for this fold
                train_loader, val_loader = create_data_loaders(
                    skin_data_dir, 
                    batch_size=64,  # Start with smaller batch size
                    num_workers=2,   # Reduce workers
                    fold_idx=fold, 
                    k_folds=k_folds
                )
                
                # Create model
                model = EfficientModel(num_classes=7)
                model = model.to(device)  # Move to GPU first
                
                # Train the model
                model, best_model_state = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_epochs=50
                )
                
                # Save fold-specific model
                save_path = f'skin_cancer_model_fold_{fold}.pth'
                torch.save({
                    'fold': fold,
                    'model_state_dict': best_model_state,
                }, save_path)
                logger.info(f"Saved model to {save_path}")
                
            except Exception as e:
                logger.error(f"Error training fold {fold + 1}: {str(e)}")
                continue
        
        # Train malaria detection model
        logger.info("Training malaria detection model...")
        malaria_data_dir = Path("backend/datasets/malaria")
        
        if not malaria_data_dir.exists():
            logger.error(f"Malaria dataset directory not found at {malaria_data_dir}")
            return
            
        if not (malaria_data_dir / "train").exists() or not any((malaria_data_dir / "train").iterdir()):
            logger.error("Malaria dataset not found or empty. Skipping training.")
            return
            
        try:
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                malaria_data_dir,
                batch_size=64,  # Start with smaller batch size
                num_workers=2    # Reduce workers
            )
            
            # Create malaria model
            model = EfficientModel(num_classes=2)
            model = model.to(device)  # Move to GPU first
            
            # Train the model
            model, best_model_state = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=50
            )
            
            # Save final model
            save_path = 'malaria_model.pth'
            torch.save({
                'model_state_dict': best_model_state,
            }, save_path)
            logger.info(f"Saved model to {save_path}")
            
        except Exception as e:
            logger.error(f"Error training malaria model: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging to file
    file_handler = logging.FileHandler('training.log')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Start training
    train_models()
