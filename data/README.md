# Medical Image Dataset

## Directory Structure

```
data/
├── raw/
│   ├── images/    # Place your medical images here
│   └── masks/     # Place corresponding segmentation masks here
└── processed/     # Processed data will be stored here (automatic)
```

## Data Preparation

1. Place your medical images (MRI, X-ray, CT scans, etc.) in the `raw/images/` directory.
2. Place the corresponding binary segmentation masks in the `raw/masks/` directory.
3. Ensure that each mask file has the same name as its corresponding image file.

## Image Format

- Supported image formats: JPG, PNG, BMP, TIFF
- Supported medical formats: NIfTI (.nii, .nii.gz)
- Images can be in grayscale or RGB
- Masks should be binary (black and white) where white (255) represents the tumor/region of interest

## Working with NIfTI Files

When using NIfTI files (.nii or .nii.gz):

1. The script automatically extracts the middle slice of 3D volumes for 2D processing
2. Masks should be binary (0 for background, non-zero values for tumor regions)
3. Ensure that your NIfTI files have proper orientation and are registered correctly
4. Both images and masks should have corresponding names (e.g., `patient1.nii` and `patient1_mask.nii`)

## Example Dataset Sources

If you don't have your own dataset, you can use publicly available medical image segmentation datasets:

1. [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (Includes NIfTI format)
2. [BRATS (Brain Tumor Segmentation Challenge)](https://www.med.upenn.edu/cbica/brats2020/data.html) (NIfTI format)
3. [ISIC (Skin Lesion Analysis)](https://challenge.isic-archive.com/data/)
4. [LUNA (Lung Nodule Analysis)](https://luna16.grand-challenge.org/Data/)
5. [Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) (Various formats including NIfTI)

Remember to preprocess these datasets to match the expected format for this project. 