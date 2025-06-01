import random
import shutil
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        images_dir (str): Path to directory containing images
        labels_dir (str): Path to directory containing labels
        output_dir (str): Path to output directory where split datasets will be saved
        train_ratio (float): Ratio of training set (default: 0.7)
        val_ratio (float): Ratio of validation set (default: 0.2)
        test_ratio (float): Ratio of test set (default: 0.1)
    """
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'valid' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'valid' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test' / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get list of image files (without extensions)
    image_files = [f.stem for f in Path(images_dir).glob('*') if f.is_file()]
    label_files = [f.stem for f in Path(labels_dir).glob('*') if f.is_file()]
    
    # Verify that each image has a corresponding label file
    valid_files = []
    for file in image_files:
        if file in label_files:
            valid_files.append(file)
        else:
            print(f"Warning: No label found for image {file}")
    
    # Shuffle files
    random.shuffle(valid_files)
    total_files = len(valid_files)
    
    # Calculate split indices
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)
    
    # Split files
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
    
    print(f"Total files: {total_files}")
    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")
    
    # Helper function to copy files
    def copy_files(files, split_name):
        for file in files:
            # Find the image file with any extension
            img_src = next(Path(images_dir).glob(f"{file}.*"))
            if img_src:
                shutil.copy(img_src, output_dir / split_name / 'images' / img_src.name)
            
            # Find the label file with any extension
            label_src = next(Path(labels_dir).glob(f"{file}.*"))
            if label_src:
                shutil.copy(label_src, output_dir / split_name / 'labels' / label_src.name)
    
    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'valid')
    copy_files(test_files, 'test')
    
    print("Dataset splitting completed successfully!")

if __name__ == "__main__":
    # Set your paths here
    images_directory = "./datasets/sardine_pose_dataset/images"  # Path to your images folder
    labels_directory = "./datasets/sardine_pose_dataset/labels"  # Path to your labels folder
    output_directory = "./datasets/sardine_pose_dataset/dataset"  # Where to save the split datasets
    
    # Split the dataset
    split_dataset(images_directory, labels_directory, output_directory)