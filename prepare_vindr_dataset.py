import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
CSV_PATH = "./data/vindr-1.0.0/breast-level_annotations.csv"
SOURCE_IMAGE_DIR = "./data/vindr-1.0.0-resized-1024/images"
OUTPUT_DIR = "./data/vindr-1.0.0-paired"
VAL_SPLIT = 0.05  # 5% for validation

def find_paired_images(df):
    """
    Find paired images (same study, same laterality, different view position).
    Returns a list of tuples (cc_row, mlo_row).
    """
    paired_images = []
    
    # Group by study_id and laterality
    grouped = df.groupby(['study_id', 'laterality'])
    
    for (study_id, laterality), group in grouped:
        # Check if we have both CC and MLO views
        cc_rows = group[group['view_position'] == 'CC']
        mlo_rows = group[group['view_position'] == 'MLO']
        
        # NOTE: This is a strict pairing rule (exactly one CC and one MLO per study/laterality).
        # Studies with multiple CC/MLO images will be skipped, which may significantly reduce data.
        if len(cc_rows) == 1 and len(mlo_rows) == 1:
            cc_row = cc_rows.iloc[0]
            mlo_row = mlo_rows.iloc[0]
            
            # Both should have the same split value
            if cc_row['split'] == mlo_row['split']:
                paired_images.append((cc_row, mlo_row))
            else:
                print(f"Warning: Mismatched splits for {study_id}-{laterality}")
    
    return paired_images

def create_output_structure():
    """Create the output directory structure."""
    for view in ['CC', 'MLO']:
        for split in ['train', 'val', 'test']:
            path = os.path.join(OUTPUT_DIR, view, split)
            os.makedirs(path, exist_ok=True)
    print(f"Created output directory structure at {OUTPUT_DIR}")

def copy_paired_images(paired_images, split_name):
    """
    Copy paired images to the appropriate directories.
    
    Args:
        paired_images: List of (cc_row, mlo_row) tuples
        split_name: 'train', 'val', or 'test'
    """
    for idx, (cc_row, mlo_row) in enumerate(tqdm(paired_images, desc=f"Copying {split_name}")):
        study_id = cc_row['study_id']
        
        # Source paths
        cc_src = os.path.join(SOURCE_IMAGE_DIR, study_id, f"{cc_row['image_id']}_resized.png")
        mlo_src = os.path.join(SOURCE_IMAGE_DIR, study_id, f"{mlo_row['image_id']}_resized.png")
        
        # Create a unique base name for the pair
        # Format: {study_id}_{laterality}_{CC/MLO}.png
        laterality = cc_row['laterality']
        base_name = f"{study_id}_{laterality}"
        
        # Destination paths
        cc_dst = os.path.join(OUTPUT_DIR, 'CC', split_name, f"{base_name}_CC.png")
        mlo_dst = os.path.join(OUTPUT_DIR, 'MLO', split_name, f"{base_name}_MLO.png")
        
        # Copy files
        try:
            if os.path.exists(cc_src):
                shutil.copy2(cc_src, cc_dst)
            else:
                print(f"Warning: Source file not found: {cc_src}")
            
            if os.path.exists(mlo_src):
                shutil.copy2(mlo_src, mlo_dst)
            else:
                print(f"Warning: Source file not found: {mlo_src}")
        except Exception as e:
            print(f"Error copying pair {base_name}: {e}")

def main():
    # Load CSV
    print(f"Loading CSV from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total images in dataset: {len(df)}")
    
    # Find paired images
    print("Finding paired images...")
    paired_images = find_paired_images(df)
    print(f"Found {len(paired_images)} paired images")
    
    # Separate by original split
    train_pairs = [(cc, mlo) for cc, mlo in paired_images if cc['split'] == 'training']
    test_pairs = [(cc, mlo) for cc, mlo in paired_images if cc['split'] == 'test']
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    
    # Split training into train and val
    if len(train_pairs) > 0:
        train_pairs, val_pairs = train_test_split(
            train_pairs, 
            test_size=VAL_SPLIT, 
            random_state=42
        )
        print(f"After split - Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    else:
        val_pairs = []
        print("No validation pairs created")
    
    # Create output structure
    create_output_structure()
    
    # Copy images
    print("\nCopying images...")
    copy_paired_images(train_pairs, 'train')
    copy_paired_images(val_pairs, 'val')
    copy_paired_images(test_pairs, 'test')
    
    print("\nDataset preparation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Total pairs: {len(train_pairs) + len(val_pairs) + len(test_pairs)}")

if __name__ == "__main__":
    main()
