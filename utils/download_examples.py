import os
import urllib.request
import argparse

def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path.
    """
    print(f"Downloading {url} to {save_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, save_path)
        print(f"Successfully downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_examples():
    """
    Download example medical images for the application.
    """
    # Define the root directory
    examples_dir = os.path.join('static', 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    # URLs for example images (these are publicly available sample medical images)
    example_files = {
        # Sample Brain MRI (JPG format)
        "brain_mri.jpg": "https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/abdominal-segmentation/sample_images/0.jpg",
        
        # Sample Lung CT (PNG format)
        "lung_ct.png": "https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/abdominal-segmentation/sample_images/5.jpg",
    }
    
    # Download each example file
    success_count = 0
    for filename, url in example_files.items():
        save_path = os.path.join(examples_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(save_path):
            print(f"{save_path} already exists. Skipping download.")
            success_count += 1
            continue
        
        # Download the file
        if download_file(url, save_path):
            success_count += 1
    
    # Print summary
    print(f"\nDownloaded {success_count} of {len(example_files)} example files.")
    print(f"Example files are stored in: {os.path.abspath(examples_dir)}")
    
    if success_count == len(example_files):
        print("\nAll example files were downloaded successfully!")
    else:
        print("\nSome files couldn't be downloaded. You can try manually downloading them.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download example medical images for the application")
    args = parser.parse_args()
    
    download_examples() 