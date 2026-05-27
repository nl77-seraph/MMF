"""
Process the OW.npz file and reorganize the data into one folder per label with one pkl file per sample.
"""

import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse

def process_npz(npz_path, output_dir):
    """

    
    Args:
        npz_path: Path to the npz file.
        output_dir: Output directory path.
    """
    # Load the npz file.
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Data example: {X[0][:4]}")  # Show the first four elements of the first sample.
    
    # Get unique labels.
    unique_labels = np.unique(y)
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Label range: {unique_labels.min()} - {unique_labels.max()}")
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Create one folder for each label.
    for label in unique_labels:
        label_dir = os.path.join(output_dir, str(int(label)))
        os.makedirs(label_dir, exist_ok=True)
    
    # Process each sample.
    print("\nProcessing data...")
    for idx in tqdm(range(len(X)), desc="Processing progress"):
        # Get the current sample.
        trace = X[idx]
        label = y[idx]
        
        # Find non-zero elements and remove padded zeros.
        non_zero_mask = trace != 0
        if np.any(non_zero_mask):
            trace = trace[non_zero_mask]
        else:
            # Skip samples that contain only zeros.
            continue
        
        # Extract timestamp and direction sequences.
        X_dir = np.sign(trace)  # Direction sequence with sign information.
        X_time = np.abs(trace)  # Timestamp information as absolute values.
        
        # Create a data dictionary compatible with the reading code.
        data_dict = {
            'time': X_time,      # Timestamp array.
            'data': X_dir,       # Direction sequence array.
            'label': int(label)  # Label.
        }
        
        # Save as a pkl file.
        label_dir = os.path.join(output_dir, str(int(label)))
        file_name = f"trace_{idx}.pkl"
        file_path = os.path.join(label_dir, file_name)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    print(f"\nProcessing complete. Processed {len(X)} samples")
    
    # Count samples for each label.
    print("\nSample statistics per label:")
    for label in unique_labels:
        label_dir = os.path.join(output_dir, str(int(label)))
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith('.pkl')])
            print(f"Label {int(label)}: {count} samples")

def verify_pkl_files(output_dir, num_samples=3):
    """
    Verify the format of generated pkl files.
    
    Args:
        output_dir: Output directory path.
        num_samples: Number of samples to verify.
    """
    print("\nVerifying generated pkl files...")
    
    # Get all label folders.
    label_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for label_dir in label_dirs[:min(num_samples, len(label_dirs))]:
        label_path = os.path.join(output_dir, label_dir)
        pkl_files = [f for f in os.listdir(label_path) if f.endswith('.pkl')]
        
        if pkl_files:
            # Read the first pkl file for verification.
            test_file = os.path.join(label_path, pkl_files[0])
            print(f"\nVerifying file: {test_file}")
            
            # Use the provided reading method.
            with open(test_file, 'rb') as f:
                raw_dict = pickle.load(f)
            
            raw_times = raw_dict['time']
            raw_datas = raw_dict['data']
            raw_labels = raw_dict['label']
            
            print(f"  Label: {raw_labels}")
            print(f"  Timestamp count: {len(raw_times)}")
            print(f"  Direction sequence count: {len(raw_datas)}")
            print(f"  Timestamp example: {raw_times[:5] if len(raw_times) >= 5 else raw_times}")
            print(f"  Direction sequence example: {raw_datas[:5] if len(raw_datas) >= 5 else raw_datas}")
            print(f"  Data length consistency check: {len(raw_times) == len(raw_datas)}")

def main():
    parser = argparse.ArgumentParser(description='Process OW.npz data files')
    parser.add_argument('--npz_path', type=str, default='/root/datasets/OW.npz', 
                        help='Path to the npz file')
    parser.add_argument('--output_dir', type=str, default='/root/datasets/OW', 
                        help='Output directory path')
    parser.add_argument('--verify', action='store_true', 
                        help='Whether to verify generated files')
    
    args = parser.parse_args()
    
    # Process data.
    process_npz(args.npz_path, args.output_dir)
    
    # Verify files.
    if args.verify:
        verify_pkl_files(args.output_dir)

if __name__ == "__main__":
    main()