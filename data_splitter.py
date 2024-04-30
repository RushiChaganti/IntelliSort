import os
import shutil
import random
from tqdm import tqdm

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.8, 0.1, 0.1)):
    # Ensure destination directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of classes (subdirectories) in the source directory
    classes = os.listdir(source_dir)

    # Iterate over each class directory
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class directories in train, val, and test directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        for dir_path in [train_class_dir, val_class_dir, test_class_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Get list of image files in the class directory
        images = os.listdir(class_dir)

        # Shuffle the list of image files
        random.shuffle(images)

        # Split the image list based on the split ratio
        num_images = len(images)
        num_train = int(num_images * split_ratio[0])
        num_val = int(num_images * split_ratio[1])

        # Copy images to train, val, and test directories with progress bars
        for i, image_name in enumerate(tqdm(images, desc=f"Splitting '{class_name}'")):
            src_path = os.path.join(class_dir, image_name)
            if i < num_train:
                dst_path = os.path.join(train_class_dir, image_name)
            elif i < num_train + num_val:
                dst_path = os.path.join(val_class_dir, image_name)
            else:
                dst_path = os.path.join(test_class_dir, image_name)
            shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    # Set paths
    source_dir = './data/'  # Directory containing all images
    train_dir = './data/train'  # Directory to store training images
    val_dir = './data/val'  # Directory to store validation images
    test_dir = './data/test'  # Directory to store test images

    # Split data
    split_data(source_dir, train_dir, val_dir, test_dir)

