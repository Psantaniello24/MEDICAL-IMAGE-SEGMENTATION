"""
Examples of how to run training with different parameter combinations.
This script provides example commands for training the U-Net model with various parameter settings.
"""

import os
import sys
import argparse
import subprocess

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def run_command(command):
    """Run a command and print its output."""
    print(f"\nRunning: {command}\n")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='Run medical image segmentation training with different parameter sets')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5], help='Example number to run (1-5)')
    parser.add_argument('--save_all_configs', action='store_true', help='Save all example configs to files')
    
    args = parser.parse_args()
    
    # Example 1: Basic training with default parameters
    example1 = "python train.py"
    
    # Example 2: Fast training with fewer epochs and quick early stopping
    example2 = "python train.py --epochs 20 --patience 5 --batch_size 4"
    
    # Example 3: Higher resolution images with more filters
    example3 = "python train.py --img_size 384 --filters 128 --batch_size 4"
    
    # Example 4: Training with slower learning rate and more patience
    example4 = "python train.py --learning_rate 1e-5 --patience 20 --epochs 200"
    
    # Example 5: Training with custom data directory
    example5 = "python train.py --data_dir custom_data/raw --img_size 256 --epochs 50"
    
    # Save configurations to files if requested
    if args.save_all_configs:
        print_header("Saving Example Configurations")
        
        if not os.path.exists('configs'):
            os.makedirs('configs')
            
        run_command(f"{example1} --save_config configs/default.json")
        run_command(f"{example2} --save_config configs/fast_training.json")
        run_command(f"{example3} --save_config configs/high_res.json")
        run_command(f"{example4} --save_config configs/slow_learning.json")
        run_command(f"{example5} --save_config configs/custom_data.json")
        
        print("\nAll configurations saved to the 'configs/' directory.")
        print("You can use them with: python train.py --load_config configs/default.json")
        return
    
    # Run a specific example if requested
    if args.example == 1:
        print_header("Example 1: Basic training with default parameters")
        print("This example uses the default parameters:")
        print("- Image size: 256x256")
        print("- Batch size: 8")
        print("- Epochs: 100")
        print("- Learning rate: 1e-4")
        run_command(example1)
        
    elif args.example == 2:
        print_header("Example 2: Fast training with fewer epochs")
        print("This example uses fewer epochs and quick early stopping:")
        print("- Epochs: 20")
        print("- Patience: 5")
        print("- Batch size: 4")
        run_command(example2)
        
    elif args.example == 3:
        print_header("Example 3: Higher resolution images with more filters")
        print("This example uses higher resolution images and more filters:")
        print("- Image size: 384x384")
        print("- Filters in first layer: 128")
        print("- Batch size: 4 (reduced due to higher memory requirements)")
        run_command(example3)
        
    elif args.example == 4:
        print_header("Example 4: Training with slower learning rate")
        print("This example uses a slower learning rate and more patience:")
        print("- Learning rate: 1e-5")
        print("- Patience: 20")
        print("- Epochs: 200")
        run_command(example4)
        
    elif args.example == 5:
        print_header("Example 5: Training with custom data directory")
        print("This example uses a custom data directory:")
        print("- Data directory: custom_data/raw")
        print("- Image size: 256x256")
        print("- Epochs: 50")
        run_command(example5)
    
    else:
        print_header("Available Examples")
        print("Choose one of the following examples to run:")
        print("Example 1: Basic training with default parameters")
        print("Example 2: Fast training with fewer epochs")
        print("Example 3: Higher resolution images with more filters")
        print("Example 4: Training with slower learning rate")
        print("Example 5: Training with custom data directory")
        print("\nUse --example <number> to run a specific example")
        print("Use --save_all_configs to save all examples to config files")

if __name__ == "__main__":
    main() 