import os
import PIL
import shutil
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from torchvision.models import vgg16
from tqdm import tqdm

# Suppress TensorFlow and other warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model(num_classes):
    model = vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Customize top layers
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 1024), nn.ReLU(inplace=True), nn.Linear(1024, num_classes)])
    model.classifier = nn.Sequential(*features)

    return model

def classify_image(image_path, transform, model):
    try:
        image = Image.open(image_path).convert('RGB')
    except (PIL.UnidentifiedImageError, OSError):
        print(f"Skipping {image_path} as it cannot be identified.")
        return None

    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item() if predicted is not None else None


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


def remove_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for folder in dirs:
            full_path = os.path.join(root, folder)
            if not os.listdir(full_path):
                os.rmdir(full_path)
        os.makedirs(root, exist_ok=True)

# Load the saved models
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
# Define the model architecture
model_people = models.vgg16(pretrained=False)  # Use VGG16 instead of ResNet50
num_ftrs = model_people.classifier[-1].in_features
model_people.classifier[-1] = nn.Linear(num_ftrs, num_classes_people)

# Load the state dictionary into the model
model_people.load_state_dict(torch.load('face_classifier.pth', map_location=torch.device('cpu')))
model_people.eval()

class_labels_objects = ['docs', 'handwritten_docs', 'People', 'pets', 'signatures']

transform_pets = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_objects = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


# Classify and sort images for objects
input_folder_objects = "./input"
output_folder_objects = "./output"
classify_images(input_folder_objects, output_folder_objects, class_labels_objects, transform_objects, model_objects)
remove_empty_folders(output_folder_objects)

# Classify and sort images for pets
input_folder_pets = 'output/pets'
output_folder_pets = 'output/pets'
classify_images_pets(input_folder_pets, output_folder_pets)
remove_empty_folders(output_folder_pets)

# Classify and sort images for people
input_folder_people = 'output/People'
output_folder_people = 'output/people'
classify_images_people(output_folder_people)
remove_empty_folders(output_folder_people)

print("Image classification and sorting complete.")
