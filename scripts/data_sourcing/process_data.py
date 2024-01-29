import os
import shutil

from sklearn.model_selection import train_test_split

# Define the path to the raw data
raw_data_path = '../../data/raw'
processed_data_path = '../../data/processed'

# Create train and test directories
train_dir = os.path.join(processed_data_path, 'train')
val_dir = os.path.join(processed_data_path, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get the list of all folders in raw data directory
folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]

# Loop over each folder and split the files into train and test sets
for folder in folders:
    # Get the list of all files in the folder
    files = os.listdir(os.path.join(raw_data_path, folder))
    print(f"Processing {folder} folder")
    print(f"Found {len(files)} images")

    if len(files) == 0:
        print(f"Skipping {folder} folder, no images found")
        continue

    corr_img = 0

    # Remove corrupted images
    for file in files:
        try:
            img = Image.open(os.path.join(raw_data_path, folder, file))
        except :
            #os.remove(os.path.join(raw_data_path, folder, file))
            #files.remove(file)
            print(f"Removed corrupted image {file}")
            corr_img += 1

    if corr_img > 0:
        print(f"Removed {corr_img} corrupted images from {folder} folder")

    print(f"Splitting {len(files)} images into train and test sets")

    # Split the files into train and test sets
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Create corresponding folder in train and test directories
    train_folder = os.path.join(train_dir, folder)
    test_folder = os.path.join(val_dir, folder)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Copy the train files to the train directory
    for file in train_files:
        shutil.copy(os.path.join(raw_data_path, folder, file), os.path.join(train_folder, file))
    
    # Copy the test files to the test directory
    for file in test_files:
        shutil.copy(os.path.join(raw_data_path, folder, file), os.path.join(test_folder, file))
