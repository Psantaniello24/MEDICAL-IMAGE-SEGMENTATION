import streamlit as st
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import tempfile
import nibabel as nib
import io  # Added this import for BytesIO
import gzip
import shutil

# Try to import SimpleITK for alternative loading
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except ImportError:
    HAVE_SITK = False
    st.warning("SimpleITK not available. For better NIFTI support, install with: pip install SimpleITK")

# Import custom modules
from src.utils import preprocess_single_image, predict_mask, overlay_mask

# Set up page configuration
st.set_page_config(
    page_title="Medical Image Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

def load_model(model_path):
    """Load the trained model"""
    try:
        # Fix imports for model loading
        from tensorflow.keras import backend as K
        
        def dice_coefficient(y_true, y_pred, smooth=1.0):
            """Dice coefficient metric for model loading"""
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            intersection = K.sum(y_true_flat * y_pred_flat)
            return (2. * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)
        
        def dice_loss(y_true, y_pred, smooth=1.0):
            """Dice loss function for model loading"""
            return 1 - dice_coefficient(y_true, y_pred, smooth)
        
        def iou_metric(y_true, y_pred, smooth=1.0):
            """IoU metric for model loading"""
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            intersection = K.sum(y_true_flat * y_pred_flat)
            union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection
            return (intersection + smooth) / (union + smooth)
        
        # First check if model exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
            
        # Load the model with custom metrics
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={
                'dice_loss': dice_loss,
                'dice_coefficient': dice_coefficient,
                'iou_metric': iou_metric
            }
        )
        return model
    except UnicodeError as e:
        st.error(f"Encoding error when loading model: {e}")
        st.error("This might be due to special characters in the model file.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def is_nifti_file(file_bytes):
    """Check if the file is a NIFTI file by examining the header"""
    # NIFTI-1 header starts with 'n+1' or 'ni1'
    if len(file_bytes) < 4:
        return False
        
    # Check for NIFTI-1 magic bytes
    if file_bytes[344:348] in [b'n+1\0', b'ni1\0']:
        return True
        
    # Check for NIFTI-2 magic bytes
    if file_bytes[0:8] == b'nii2\x00\r\n\x1a\n':
        return True
        
    # Check for gzipped files
    if file_bytes[0:2] == b'\x1f\x8b':
        return True
        
    return False

def uncompress_gzip(input_path, output_path):
    """Uncompress a gzipped file"""
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_path

def load_nifti_image(uploaded_file):
    """Load a medical image file, with special handling for NIfTI files"""
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Check if the file is gzipped and try to uncompress
        if uploaded_file.type == 'application/x-gzip' or uploaded_file.name.endswith('.gz'):
            try:
                uncompressed_path = tmp_path + '_uncompressed'
                uncompress_gzip(tmp_path, uncompressed_path)
                
                # Also try with .nii extension for better compatibility
                uncompressed_nii_path = uncompressed_path + '.nii'
                shutil.copy(uncompressed_path, uncompressed_nii_path)
                
                # Keep track of all temp files to clean up later
                temp_files = [tmp_path, uncompressed_path, uncompressed_nii_path]
                
                # Try both paths
                paths_to_try = [uncompressed_path, uncompressed_nii_path, tmp_path]
            except Exception:
                paths_to_try = [tmp_path]
                temp_files = [tmp_path]
        else:
            paths_to_try = [tmp_path]
            temp_files = [tmp_path]
        
        # Try multiple loading methods on each path
        img_data = None
        loading_errors = []
        
        for path in paths_to_try:
            if img_data is not None:
                break
                
            # Approach 1: Try using nibabel directly
            try:
                nii_img = nib.load(path)
                img_data = nii_img.get_fdata()
                break
            except Exception as e:
                loading_errors.append(f"Nibabel: {str(e)}")
                
            # Approach 2: Try using SimpleITK if available
            if HAVE_SITK:
                try:
                    itk_img = sitk.ReadImage(path)
                    img_data = sitk.GetArrayFromImage(itk_img)
                    
                    # SimpleITK loads with different axis ordering
                    if len(img_data.shape) == 3:
                        img_data = np.transpose(img_data, (1, 2, 0))
                    
                    break
                except Exception as e:
                    loading_errors.append(f"SimpleITK: {str(e)}")
            
            # Approach 3: For gzipped files, try manual handling
            if path.endswith('.gz'):
                try:
                    with gzip.open(path, 'rb') as f:
                        content = f.read()
                    
                    # Try to detect if it's a raw image format
                    if len(content) > 348 and content[344:348] in [b'n+1\0', b'ni1\0']:
                        # Write to a temporary file and load with nibabel
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as raw_file:
                            raw_file.write(content)
                            raw_path = raw_file.name
                        
                        temp_files.append(raw_path)
                        nii_img = nib.load(raw_path)
                        img_data = nii_img.get_fdata()
                        break
                except Exception as e:
                    loading_errors.append(f"Gzip extraction: {str(e)}")
        
        if img_data is None:
            # All methods failed
            raise ValueError(f"Could not load the file with any available method. Errors: {'; '.join(loading_errors)}")
        
        # Display information about the loaded file
        st.success(f"Image loaded successfully. Shape: {img_data.shape}, Data type: {img_data.dtype}")
        
        # Select appropriate slice for visualization
        if len(img_data.shape) == 3:
            total_slices = img_data.shape[2]
            slice_idx = img_data.shape[2] // 2
            
            # Add a slider to select the slice if there are multiple slices
            if total_slices > 1:
                slice_idx = st.slider("Select slice", 0, total_slices-1, slice_idx)
                
            img = img_data[:, :, slice_idx]
        elif len(img_data.shape) == 4:
            # Handle 4D data (time series)
            total_slices = img_data.shape[2]
            total_timepoints = img_data.shape[3]
            
            slice_idx = img_data.shape[2] // 2
            time_idx = 0
            
            # Add sliders to select slice and timepoint
            col1, col2 = st.columns(2)
            with col1:
                slice_idx = st.slider("Select slice", 0, total_slices-1, slice_idx)
            with col2:
                time_idx = st.slider("Select timepoint", 0, total_timepoints-1, time_idx)
                
            img = img_data[:, :, slice_idx, time_idx]
        else:
            img = img_data
        
        # Normalize to [0, 255] if not already in that range
        if img.max() > 0:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # Convert to RGB for display
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        
        return img_rgb
    except Exception as e:
        st.error(f"Error loading file: {e}")
        
        # Fallback option - offer to load as a standard image
        if uploaded_file.type.startswith('image/'):
            st.warning("The file appears to be a standard image. Trying to load it as a regular image...")
            try:
                image = Image.open(uploaded_file)
                return np.array(image)
            except Exception as img_error:
                st.error(f"Failed to load as standard image: {img_error}")
        
        # Add more specific guidance based on file type
        if uploaded_file.type == 'application/x-gzip':
            st.info(
                "Your file appears to be gzipped but couldn't be loaded as a NIFTI file. "
                "This could happen if:\n"
                "1. The file is not actually a NIFTI file (just a regular gzipped file)\n"
                "2. The file uses a non-standard format or is corrupted\n\n"
                "Try using a tool like MRIcroGL to convert your file to a standard format."
            )
        
        return None

def get_prediction(model, image):
    """Get segmentation prediction for an image"""
    # Preprocess image
    img_processed = preprocess_single_image(image)
    
    # Get prediction
    mask = model.predict(img_processed)
    
    # Overlay mask on original image
    img_with_mask = overlay_mask(img_processed, mask, alpha=0.5, color=[1, 0, 0])
    
    # Get binary mask for separate display
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    return img_processed, binary_mask, img_with_mask

def convert_to_display_format(img):
    """Convert image to displayable format"""
    # Remove batch dimension if present
    if len(img.shape) == 4:
        img = np.squeeze(img, axis=0)
    
    # Convert to uint8 for display
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(np.squeeze(img, axis=2), cv2.COLOR_GRAY2RGB)
    
    return img

def main():
    st.title("Medical Image Tumor Segmentation")
    st.write("Upload a medical image for tumor segmentation using a U-Net model.")
    
    # Add notice about SimpleITK
    if not HAVE_SITK:
        st.warning(
            "For better support of medical image formats like NIFTI, please install SimpleITK:\n"
            "```\npip install SimpleITK\n```"
        )
    
    # Check if model exists
    model_path = os.path.join('models', 'unet_model_final.h5')
    if not os.path.exists(model_path):
        st.error(
            "Model not found. Please train the model first by running: \n\n"
            "```python train.py```"
        )
        st.info(
            "If you don't have a dataset yet, place your medical images in the 'data/raw/images' "
            "directory and the corresponding masks in 'data/raw/masks'."
        )
        return
    
    # Load the model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    if model is None:
        return
    
    st.success("Model loaded successfully!")
    
    # Create sidebar
    st.sidebar.title("Settings")
    threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Add example images section
    st.sidebar.subheader("Example Images")
    example_option = st.sidebar.selectbox(
        "Use example image", 
        ["None", "Brain MRI (JPG)", "Lung CT (PNG)"]
    )
    
    # Set up file uploader
    if example_option == "None":
        # File uploader with expanded medical image support
        uploaded_file = st.file_uploader(
            "Choose a medical image...", 
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "nii", "nii.gz", "dcm", "ima"]
        )
    else:
        # Use example image
        uploaded_file = None
        example_image_path = None
        
        # Check if example directory exists
        if not os.path.exists('static/examples'):
            os.makedirs('static/examples', exist_ok=True)
            st.info("Creating example images directory. Please upload example images manually.")
        else:
            # Map selected option to file path
            if example_option == "Brain MRI (JPG)":
                example_image_path = "static/examples/brain_mri.jpg"
                if not os.path.exists(example_image_path):
                    st.warning(f"Example image not found at {example_image_path}. Please add it manually.")
                    example_image_path = None
            elif example_option == "Lung CT (PNG)":
                example_image_path = "static/examples/lung_ct.png"
                if not os.path.exists(example_image_path):
                    st.warning(f"Example image not found at {example_image_path}. Please add it manually.")
                    example_image_path = None
            
            if example_image_path:
                st.info(f"Using example image: {example_option}")
    
    # Process the image - either uploaded or example
    image_to_process = None
    
    # Handle example image if selected
    if example_option != "None" and example_image_path and os.path.exists(example_image_path):
        try:
            image_array = cv2.imread(example_image_path)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Show original image
            st.subheader("Original Example Image")
            st.image(image_array, width=400)
            
            image_to_process = image_array
        except Exception as e:
            st.error(f"Error loading example image: {e}")
    
    # Handle uploaded file if any
    elif uploaded_file is not None:
        st.write(f"File name: {uploaded_file.name}")
        st.write(f"File type: {uploaded_file.type}")
        
        # Load the image data
        with st.spinner("Loading medical image file..."):
            # For DICOM and NIFTI files, use our specialized loader
            if uploaded_file.name.lower().endswith(('.nii', '.nii.gz', '.dcm', '.ima')):
                image_array = load_nifti_image(uploaded_file)
                if image_array is None:
                    st.error(f"Failed to load the medical image file: {uploaded_file.name}")
                    st.error("Please try a different file or format.")
                    st.stop()
                
                # Show original image
                st.subheader("Original Image")
                st.image(image_array, width=400)
                
                image_to_process = image_array
            else:
                # For standard image formats
                try:
                    image = Image.open(uploaded_file)
                    image_array = np.array(image)
                    
                    # Check if the image is actually loaded
                    if image_array.size == 0:
                        st.error("Could not read image data from the uploaded file.")
                        st.stop()
                    
                    # Show original image
                    st.subheader("Original Image")
                    st.image(image, width=400)
                    
                    image_to_process = image_array
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    st.error("Please make sure your file is a valid image file.")
                    st.stop()
    
    # Process the image if available
    if image_to_process is not None:
        # Get predictions
        with st.spinner("Generating segmentation..."):
            try:
                img_processed, binary_mask, img_with_mask = get_prediction(model, image_to_process)
                
                # Convert results for display
                binary_mask_display = convert_to_display_format(binary_mask)
                img_with_mask_display = convert_to_display_format(img_with_mask)
                
                # Create columns for display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tumor Segmentation Mask")
                    st.image(binary_mask_display, width=400)
                
                with col2:
                    st.subheader("Overlay: Image + Segmentation")
                    st.image(img_with_mask_display, width=400)
                
                # Segmentation metrics (simple calculations for demo purposes)
                num_pixels = binary_mask.size
                num_tumor_pixels = np.sum(binary_mask)
                tumor_percentage = (num_tumor_pixels / num_pixels) * 100
                
                st.subheader("Segmentation Analysis")
                st.write(f"- Threshold used: {threshold:.2f}")
                st.write(f"- Tumor area: {num_tumor_pixels} pixels")
                st.write(f"- Percentage of image area: {tumor_percentage:.2f}%")
                
                # Add disclaimer
                st.info(
                    "⚠️ **Disclaimer**: This is a demonstration only and should not be used for actual medical diagnosis. "
                    "The predictions are not clinically validated."
                )
                
            except Exception as e:
                st.error(f"Error processing the image: {e}")
                st.error("Please make sure your image is valid and try again.")
    
    # Add information about the model 
    st.sidebar.subheader("About the Model")
    st.sidebar.write(
        "This application uses a U-Net architecture trained for medical image segmentation. "
        "The model is trained to detect tumors in medical images."
    )
    
    # Add dataset information
    st.sidebar.subheader("Dataset")
    st.sidebar.write(
        "The model was trained on a dataset of medical images and their corresponding segmentation masks."
    )
    
    # Add information about supported formats
    st.sidebar.subheader("Supported Formats")
    st.sidebar.write(
        "This application supports the following file formats:\n"
        "- Standard image formats: JPG, PNG, BMP, TIFF\n"
        "- Medical imaging formats: NIfTI (.nii, .nii.gz)\n"
        "- DICOM formats: .dcm, .ima (with SimpleITK)"
    )

    # Add troubleshooting tips
    st.sidebar.subheader("Troubleshooting")
    st.sidebar.write(
        "If you're having trouble loading NIFTI files:\n"
        "1. Install SimpleITK: `pip install SimpleITK`\n"
        "2. Try converting your files to a different format\n"
        "3. Make sure your NIFTI files are properly formatted\n"
        "4. For DICOM series, try converting to NIFTI first"
    )
    
    # Add format conversion tips
    st.sidebar.subheader("Format Conversion")
    st.sidebar.write(
        "Tools for medical image conversion:\n"
        "1. [MRIcroGL](https://www.nitrc.org/projects/mricrogl)\n"
        "2. [3D Slicer](https://www.slicer.org/)\n"
        "3. [ITK-SNAP](http://www.itksnap.org/)\n"
        "4. Python libraries: nibabel, SimpleITK"
    )

if __name__ == "__main__":
    main() 