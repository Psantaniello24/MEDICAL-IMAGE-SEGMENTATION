import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.backend as K

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Dice coefficient metric.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss function.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def iou_metric(y_true, y_pred, smooth=1.0):
    """
    IoU (Intersection over Union) metric.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection
    return (intersection + smooth) / (union + smooth)

def build_unet(input_size=(256, 256, 3), n_filters=64, dropout=0.2, batch_norm=True):
    """
    Build U-Net model for medical image segmentation.
    
    Args:
        input_size: Input image size
        n_filters: Number of filters in first layer
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        
    Returns:
        U-Net model
    """
    # Input
    inputs = Input(input_size)
    
    # Contracting path (encoder)
    # Block 1
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if dropout > 0:
        pool1 = Dropout(dropout)(pool1)
    
    # Block 2
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if dropout > 0:
        pool2 = Dropout(dropout)(pool2)
    
    # Block 3
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    if batch_norm:
        conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    if batch_norm:
        conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if dropout > 0:
        pool3 = Dropout(dropout)(pool3)
    
    # Block 4
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    if batch_norm:
        conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    if batch_norm:
        conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if dropout > 0:
        pool4 = Dropout(dropout)(pool4)
    
    # Bridge
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    if batch_norm:
        conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(n_filters*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    if batch_norm:
        conv5 = BatchNormalization()(conv5)
    
    # Expansive path (decoder)
    # Block 6
    up6 = Conv2D(n_filters*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    if dropout > 0:
        merge6 = Dropout(dropout)(merge6)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    if batch_norm:
        conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(n_filters*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    if batch_norm:
        conv6 = BatchNormalization()(conv6)
    
    # Block 7
    up7 = Conv2D(n_filters*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    if dropout > 0:
        merge7 = Dropout(dropout)(merge7)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    if batch_norm:
        conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(n_filters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    if batch_norm:
        conv7 = BatchNormalization()(conv7)
    
    # Block 8
    up8 = Conv2D(n_filters*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    if dropout > 0:
        merge8 = Dropout(dropout)(merge8)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    if batch_norm:
        conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(n_filters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    if batch_norm:
        conv8 = BatchNormalization()(conv8)
    
    # Block 9
    up9 = Conv2D(n_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    if dropout > 0:
        merge9 = Dropout(dropout)(merge9)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    if batch_norm:
        conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    if batch_norm:
        conv9 = BatchNormalization()(conv9)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with appropriate loss and metrics.
    
    Args:
        model: U-Net model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    # Compile with combined loss (binary crossentropy + dice loss)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=dice_loss,
        metrics=['binary_accuracy', dice_coefficient, iou_metric]
    )
    
    return model 