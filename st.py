import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
import shutil
import torch
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm

# Function to build the model
def build_model(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Customize top layers
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 1024), nn.ReLU(inplace=True), nn.Linear(1024, num_classes)])
    model.classifier = nn.Sequential(*features)

    return model

# Function to classify images
def classify_image(image_path, transform, model):
    try:
        image = Image.open(image_path).convert('RGB')
    except (UnidentifiedImageError, OSError):  # Handle UnidentifiedImageError
        print(f"Skipping {image_path} as it cannot be identified.")
        return None

    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item() if predicted is not None else None

# Function to classify and move images
def classify_images(input_folder, output_folder, class_labels, transform, model):
    os.makedirs(output_folder, exist_ok=True)

    # Wrap the loop with tqdm for progress tracking
    for filename in tqdm(os.listdir(input_folder), desc="Classifying images"):
        image_path = os.path.join(input_folder, filename)
        if os.path.isfile(image_path):
            predicted_class = classify_image(image_path, transform, model)
            if predicted_class is not None:
                class_label = class_labels[predicted_class]
                class_folder = os.path.join(output_folder, class_label)
                os.makedirs(class_folder, exist_ok=True)

                shutil.move(image_path, os.path.join(class_folder, filename))


transform_pets = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to classify images and move them to respective folders for pets
def classify_image_pets(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_pets(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_pets(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item(), image_path

def classify_images_pets(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder), desc="Classifying pet images"):
        image_path = os.path.join(input_folder, filename)
        if os.path.isfile(image_path):
            predicted_class, image_path = classify_image_pets(image_path)

            breed_folder = os.path.join(output_folder, str(predicted_class))
            os.makedirs(breed_folder, exist_ok=True)

            shutil.move(image_path, os.path.join(breed_folder, filename))


def classify_images_people(new_images_dir):
    def classify_image(image_path):
        image = Image.open(image_path)
        image_tensor = transform_objects(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model_people(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    for i in range(num_classes_people):
        folder_path = os.path.join(new_images_dir, f'class_{i}')
        os.makedirs(folder_path, exist_ok=True)

    for filename in tqdm(os.listdir(new_images_dir), desc="Classifying people images"):
        image_path = os.path.join(new_images_dir, filename)
        if os.path.isfile(image_path):
            predicted_class = classify_image(image_path)
            destination_folder = os.path.join(new_images_dir, f'class_{predicted_class}')
            shutil.move(image_path, os.path.join(destination_folder, filename))

def remove_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for folder in dirs:
            full_path = os.path.join(root, folder)
            if not os.listdir(full_path):
                os.rmdir(full_path)
        os.makedirs(root, exist_ok=True)

# Define the models and transformations
num_classes_pets = 37
model_pets = models.resnet18(pretrained=False)
model_pets.fc = nn.Linear(model_pets.fc.in_features, num_classes_pets)
model_pets.load_state_dict(torch.load('pets.pth'))
model_pets.eval()

num_classes_objects = 5
model_objects = build_model(num_classes_objects)
model_objects.load_state_dict(torch.load("top.pth"))
model_objects.eval()

num_classes_people = 53
model_people = models.vgg16(pretrained=False)
num_ftrs = model_people.classifier[-1].in_features
model_people.classifier[-1] = nn.Linear(num_ftrs, num_classes_people)
model_people.load_state_dict(torch.load('face_classifier.pth', map_location=torch.device('cpu')))
model_people.eval()

class_labels_objects = ['Docs', 'Handwritten Docs', 'People', 'Pets', 'Signatures']

transform_objects = transforms.Compose([
    transforms.Resize((500, 500)),  # Adjust size for better viewing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def remove_empty_folders(output_folder):
    for root, dirs, files in os.walk(output_folder, topdown=False):
        for folder in dirs:
            full_path = os.path.join(root, folder)
            if not os.listdir(full_path):
                os.rmdir(full_path)
def display_images(folder_path):
    st.subheader(f"Images in {os.path.basename(folder_path)}")
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                images.append(image_path)

    # Display images in a grid with 5 images per row
    num_images = len(images)
    images_per_row = 5
    if num_images > 0:
        num_rows = (num_images + images_per_row - 1) // images_per_row
        columns = st.columns(images_per_row)
        for i in range(num_rows):
            for j in range(images_per_row):
                if i * images_per_row + j < num_images:
                    image_path = images[i * images_per_row + j]
                    columns[j].image(image_path, width=150)
    remove_empty_folders("./output") 
def main():
    # Check if session_state exists, if not, initialize it
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {}

    st.title("Image Gallery")

    # Input folder selection
    input_folder = st.text_input("Enter the path of the input folder:", "./input")

    if not os.path.exists(input_folder):
        st.error("Input folder not found!")
        return

    # Button to start classification and sorting
    if st.button("Classify and Sort Images"):
        classify_images(input_folder, "./output", class_labels_objects, transform_objects, model_objects)
        remove_empty_folders("./output")
        input_folder_pets = 'output/pets'
        output_folder_pets = 'output/pets'
        output_folder_people = 'output/people'
        classify_images_people(output_folder_people)
        classify_images_pets(input_folder_pets, output_folder_pets)
        remove_empty_folders("./output") 
        st.success("Image classification and sorting completed.")
        st.session_state.session_state["classification_completed"] = True  # Set flag to True when sorting is done

    st.title("Sorted Images")
    st.subheader("Click below to view the images")

    # Initialize class_buttons only if classification is done
    if st.session_state.get("classification_completed", True):  # Check if flag exists and is True
        class_buttons = {}
        for label in class_labels_objects:
            class_buttons[label] = st.button(label)

        # Display images based on button clicks
        for label, button in class_buttons.items():
            if button:
                class_folder = os.path.join("./output", label)
                display_images(class_folder)
if __name__ == "__main__":
    main()

