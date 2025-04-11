import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nibabel as nib

def load_data(img_dir, mask_dir, img_size=(256, 256)):
    """
    Load images and masks from directories, resize them to the specified size.
    Supports both standard image formats and .nii files.
    
    Args:
        img_dir: Directory containing images
        mask_dir: Directory containing corresponding masks
        img_size: Target size for resizing (height, width)
        
    Returns:
        X: Array of resized images
        y: Array of resized masks
    """
    images = []
    masks = []
    
    image_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
        # Load image based on file extension
        img_path = os.path.join(img_dir, img_file)
        
        if img_file.endswith('.nii') or img_file.endswith('.nii.gz'):
            # Load NIfTI image
            try:
                nii_img = nib.load(img_path)
                # Get middle slice of 3D volume for 2D processing
                img_data = nii_img.get_fdata()
                
                # For 3D volumes, extract middle slice
                if len(img_data.shape) == 3:
                    slice_idx = img_data.shape[2] // 2
                    img = img_data[:, :, slice_idx]
                else:
                    img = img_data
                
                # Normalize to [0, 255] if not already in that range
                if img.max() > 0:
                    img = (img / img.max() * 255).astype(np.uint8)
                
                # If the image is single channel, convert to 3 channels
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
            except Exception as e:
                print(f"Failed to load NIfTI image: {img_path}. Error: {e}")
                continue
        else:
            # Load standard image formats
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            # Convert to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask based on file extension
        mask_path = os.path.join(mask_dir, mask_file)
        
        if mask_file.endswith('.nii') or mask_file.endswith('.nii.gz'):
            # Load NIfTI mask
            try:
                nii_mask = nib.load(mask_path)
                mask_data = nii_mask.get_fdata()
                
                # For 3D volumes, extract middle slice
                if len(mask_data.shape) == 3:
                    slice_idx = mask_data.shape[2] // 2
                    mask = mask_data[:, :, slice_idx]
                else:
                    mask = mask_data
                
                # Ensure mask is binary (0 or 1)
                mask = (mask > 0).astype(np.uint8) * 255
            except Exception as e:
                print(f"Failed to load NIfTI mask: {mask_path}. Error: {e}")
                continue
        else:
            # Load standard image formats
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue
        
        # Resize
        img = cv2.resize(img, img_size)
        mask = cv2.resize(mask, img_size)
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Add to lists
        images.append(img)
        masks.append(mask)
    
    if len(images) == 0:
        raise ValueError("No valid image/mask pairs found. Please check your data directory and file formats.")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(masks)
    
    # Add channel dimension to masks if needed
    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)
    
    return X, y

def preprocess_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation and test sets, and normalize images.
    
    Args:
        X: Array of images
        y: Array of masks
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Normalized and split datasets
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation set from training set
    val_proportion_of_train_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_proportion_of_train_val, 
        random_state=random_state
    )
    
    # Normalize images to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Ensure masks are binary
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_data_generator(X_train, y_train, batch_size=8, seed=42):
    """
    Create data generator with augmentation for training.
    
    Args:
        X_train: Training images
        y_train: Training masks
        batch_size: Batch size
        seed: Random seed for reproducibility
        
    Returns:
        Data generator
    """
    # Image data generator with augmentation
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create generators
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Fit generators on data if needed (not required for most augmentations)
    # image_datagen.fit(X_train, augment=True, seed=seed)
    # mask_datagen.fit(y_train, augment=True, seed=seed)
    
    # Set up image generators
    image_generator = image_datagen.flow(
        X_train, 
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )
    
    mask_generator = mask_datagen.flow(
        y_train, 
        batch_size=batch_size,
        seed=seed,
        shuffle=True
    )
    
    # Create a generator that yields (batch of images, batch of masks)
    def combined_generator():
        while True:
            X_batch = image_generator.next()
            y_batch = mask_generator.next()
            yield X_batch, y_batch
    
    return combined_generator()

def test_generator(generator, num_batches=1):
    """
    Test the generator by getting a few batches.
    
    Args:
        generator: The data generator
        num_batches: Number of batches to test
        
    Returns:
        Examples from the generator
    """
    examples = []
    for i in range(num_batches):
        x_batch, y_batch = next(generator)
        examples.append((x_batch, y_batch))
        print(f"Batch {i+1}: X shape: {x_batch.shape}, y shape: {y_batch.shape}")
    return examples 