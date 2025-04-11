import os
import cv2
import numpy as np
import argparse

def create_example_brain_mri():
    """
    Create a synthetic brain MRI example image with a simulated tumor.
    """
    # Create a 512x512 black background
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Create the brain outline (gray ellipse)
    cv2.ellipse(img, (256, 256), (180, 220), 0, 0, 360, (150, 150, 150), -1)
    
    # Create the inner brain structure (lighter gray)
    cv2.ellipse(img, (256, 256), (160, 200), 0, 0, 360, (200, 200, 200), -1)
    
    # Add a simulated tumor (red/bright area)
    cv2.circle(img, (320, 200), 40, (240, 120, 120), -1)
    
    # Add some noise to make it look more realistic
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some structure details (ventricles)
    cv2.ellipse(img, (210, 250), (20, 40), 30, 0, 360, (100, 100, 100), -1)
    cv2.ellipse(img, (310, 250), (20, 40), -30, 0, 360, (100, 100, 100), -1)
    
    # Add some text to indicate this is a sample
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "EXAMPLE MRI - NOT REAL PATIENT DATA", (30, 30), font, 0.7, (255, 255, 255), 2)
    
    return img

def create_example_lung_ct():
    """
    Create a synthetic lung CT scan example with a simulated nodule.
    """
    # Create a 512x512 dark background (representing body)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 40
    
    # Create the chest cavity outline
    cv2.ellipse(img, (256, 256), (220, 180), 0, 0, 360, (70, 70, 70), -1)
    
    # Create left lung
    cv2.ellipse(img, (180, 230), (80, 130), 15, 0, 360, (200, 200, 200), -1)
    
    # Create right lung
    cv2.ellipse(img, (330, 230), (80, 130), -15, 0, 360, (200, 200, 200), -1)
    
    # Add a simulated nodule in the right lung
    cv2.circle(img, (350, 200), 20, (120, 120, 240), -1)
    
    # Add some thoracic structure (spine, etc.)
    cv2.rectangle(img, (240, 160), (270, 350), (100, 100, 100), -1)
    
    # Add ribs
    for i in range(5):
        y_pos = 170 + i * 40
        # Left ribs
        cv2.ellipse(img, (256, y_pos), (220, 15), 0, 130, 180, (100, 100, 100), 4)
        # Right ribs
        cv2.ellipse(img, (256, y_pos), (220, 15), 0, 0, 50, (100, 100, 100), 4)
    
    # Add some noise to make it look more realistic
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some text to indicate this is a sample
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "EXAMPLE CT - NOT REAL PATIENT DATA", (30, 30), font, 0.7, (255, 255, 255), 2)
    
    return img

def create_examples():
    """
    Create example medical images for the application.
    """
    # Define the root directory
    examples_dir = os.path.join('static', 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    print("Creating example medical images...")
    
    # Create and save the brain MRI example
    brain_mri_path = os.path.join(examples_dir, "brain_mri.jpg")
    if not os.path.exists(brain_mri_path):
        brain_mri = create_example_brain_mri()
        cv2.imwrite(brain_mri_path, brain_mri)
        print(f"Created brain MRI example: {brain_mri_path}")
    else:
        print(f"Brain MRI example already exists: {brain_mri_path}")
    
    # Create and save the lung CT example
    lung_ct_path = os.path.join(examples_dir, "lung_ct.png")
    if not os.path.exists(lung_ct_path):
        lung_ct = create_example_lung_ct()
        cv2.imwrite(lung_ct_path, lung_ct)
        print(f"Created lung CT example: {lung_ct_path}")
    else:
        print(f"Lung CT example already exists: {lung_ct_path}")
    
    print(f"\nExample files are stored in: {os.path.abspath(examples_dir)}")
    print("All example images created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create example medical images for the application")
    args = parser.parse_args()
    
    create_examples() 