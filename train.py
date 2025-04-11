import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import json

# Fix TensorFlow imports
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Import custom modules
from src.data_preprocessing import load_data, preprocess_data, create_data_generator
from src.model import build_unet, compile_model
from src.utils import plot_training_history, visualize_predictions, evaluate_model, save_model_summary

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def save_config(args, filename='config.json'):
    """Save training configuration to a JSON file."""
    config = vars(args)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filename}")

def load_config(filename='config.json'):
    """Load training configuration from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filename}")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def main(args):
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save configuration if requested
    if args.save_config:
        save_config(args, args.save_config)
    
    # Configuration
    IMG_SIZE = (args.img_size, args.img_size)
    INPUT_SHAPE = (*IMG_SIZE, 3)  # RGB images
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    print(f"\n=== Training Configuration ===")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Filters (first layer): {args.filters}")
    print(f"Dropout rate: {args.dropout}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.val_size}")
    print(f"Data directory: {args.data_dir}")
    print(f"==============================\n")
    
    # Load data
    print("Loading data...")
    images_dir = os.path.join(args.data_dir, 'images')
    masks_dir = os.path.join(args.data_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Error: Data directories not found.")
        print(f"Expected: {images_dir} and {masks_dir}")
        print("Please place your medical image dataset as described in the README.")
        return
    
    try:
        X, y = load_data(images_dir, masks_dir, img_size=IMG_SIZE)
        print(f"Loaded {X.shape[0]} images with shape {X.shape[1:]} and {y.shape[0]} masks with shape {y.shape[1:]}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("Please check that your data is in the correct format and try again.")
        return
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data(
            X, y, test_size=args.test_size, val_size=args.val_size
        )
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return
    
    # Create data generator with augmentation
    print("Creating data generator with augmentation...")
    train_generator = create_data_generator(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Test the generator to make sure it works
    try:
        print("Testing data generator...")
        from src.data_preprocessing import test_generator
        test_generator(train_generator, num_batches=1)
        print("Data generator is working correctly!")
    except Exception as e:
        print(f"Error with data generator: {e}")
        print("Trying an alternative approach...")
        # Alternative approach without generator
        print("Will train without data augmentation")
        use_generator = False
    else:
        use_generator = True
    
    # Build and compile model
    print("Building U-Net model...")
    model = build_unet(
        input_size=INPUT_SHAPE, 
        n_filters=args.filters, 
        dropout=args.dropout, 
        batch_norm=True
    )
    
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Save model summary - wrapped in try/except to continue even if this fails
    try:
        save_model_summary(model, file_path='models/model_summary.txt')
    except Exception as e:
        print(f"Warning: Failed to save model summary: {e}")
        print("Continuing with training...")
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join('models', 'model_best.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coefficient', 
            patience=args.patience, 
            verbose=1, 
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coefficient', 
            factor=0.1, 
            patience=5, 
            verbose=1, 
            mode='max', 
            min_lr=1e-7
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Train model
    print(f"Training model for {EPOCHS} epochs...")
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    
    if use_generator:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Train without generator (fallback option)
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    model.load_weights(os.path.join('models', 'model_best.h5'))
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Visualize predictions
    print("Generating visualization of predictions...")
    visualize_predictions(model, X_test, y_test, num_samples=min(4, len(X_test)), save_path='predictions.png')
    
    # Save final model
    print("Saving model...")
    model.save(os.path.join('models', 'unet_model_final.h5'))
    
    print("Training completed!")
    print(f"Final model saved to 'models/unet_model_final.h5'")
    print(f"Visualization saved to 'predictions.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net model for medical image segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Directory containing images and masks folders')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for resizing')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    
    # Model parameters
    parser.add_argument('--filters', type=int, default=64, help='Number of filters in first layer of U-Net')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Dataset split parameters
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of training data for validation')
    
    # Configuration file options
    parser.add_argument('--load_config', type=str, help='Load parameters from a config file')
    parser.add_argument('--save_config', type=str, help='Save current parameters to a config file')
    
    args = parser.parse_args()
    
    # Load config file if specified
    if args.load_config:
        config = load_config(args.load_config)
        if config:
            # Update args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    main(args) 