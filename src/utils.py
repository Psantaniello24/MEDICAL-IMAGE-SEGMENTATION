import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history
    """
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot metrics
    plt.subplot(122)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def visualize_predictions(model, test_images, test_masks, num_samples=4, save_path='predictions.png'):
    """
    Visualize model predictions.
    
    Args:
        model: Trained model
        test_images: Test images
        test_masks: Test masks
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    # Get predictions
    predictions = model.predict(test_images[:num_samples])
    
    # Create figure
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(test_images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(np.squeeze(test_masks[i]), cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(np.squeeze(predictions[i]), cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_images, test_masks):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_images: Test images
        test_masks: Test masks
        
    Returns:
        Evaluation metrics
    """
    # Evaluate model
    results = model.evaluate(test_images, test_masks, verbose=1)
    
    # Print results
    print(f'Loss: {results[0]:.4f}')
    print(f'Binary Accuracy: {results[1]:.4f}')
    print(f'Dice Coefficient: {results[2]:.4f}')
    print(f'IoU: {results[3]:.4f}')
    
    return results

def preprocess_single_image(image_path, target_size=(256, 256)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to image
        target_size: Target image size
        
    Returns:
        Preprocessed image
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # If it's already a numpy array
        img = image_path
        if len(img.shape) == 2:  # If grayscale, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # If RGBA, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_mask(model, image, threshold=0.5):
    """
    Predict mask for a single image.
    
    Args:
        model: Trained model
        image: Input image (preprocessed)
        threshold: Threshold for binarization
        
    Returns:
        Predicted mask
    """
    # Make prediction
    prediction = model.predict(image)
    
    # Threshold
    prediction = (prediction > threshold).astype(np.uint8)
    
    return prediction

def overlay_mask(image, mask, alpha=0.5, color=[1, 0, 0]):
    """
    Overlay mask on image.
    
    Args:
        image: Original image
        mask: Predicted mask
        alpha: Transparency
        color: Mask color (RGB)
        
    Returns:
        Image with mask overlay
    """
    # Remove batch dimension
    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)
    if len(mask.shape) == 4:
        mask = np.squeeze(mask, axis=0)
    
    # Remove channel dimension from mask if it exists
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask, axis=2)
    
    # Create RGB mask
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for i in range(3):
        mask_rgb[:, :, i] = mask * color[i]
    
    # Overlay
    overlay = image * (1 - alpha) + mask_rgb * alpha
    
    # Clip values
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def save_model_summary(model, file_path='model_summary.txt'):
    """
    Save model summary to file.
    
    Args:
        model: Model
        file_path: Path to save the summary
    """
    # Get model summary
    from io import StringIO
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    
    # Save to file with UTF-8 encoding
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(summary_string.getvalue())
        print(f'Model summary saved to {file_path}')
    except Exception as e:
        print(f"Warning: Could not save model summary due to: {e}")
        print("Continuing with training...") 