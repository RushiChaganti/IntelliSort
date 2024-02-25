from PIL import Image
import os

def resize_images(directory_path, output_size=(100, 100)):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img_resized = img.resize(output_size, Image.LANCZOS)
                    img_resized.save(file_path)
                    print(f"Resized: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Specify the path to your main directory
main_directory = r".\data"
# Specify the output size for the images
output_size = (100, 100)

# Call the function to resize images in subdirectories
resize_images(main_directory, output_size)
