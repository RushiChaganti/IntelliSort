import os
from shutil import move
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load the image classification model
model = load_model("intellisort.h5")

input_folder_path = r".\input"
output_folder_path = r".\output"
class_labels = ['docs', 'handwritten_docs', 'People', 'pets', 'signatures']

def predict_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions)

    return class_labels[predicted_class_index]

def process_image(filename):
    image_path = os.path.join(input_folder_path, filename)

    predicted_label = predict_class(image_path)

    output_class_folder = os.path.join(output_folder_path, predicted_label)
    os.makedirs(output_class_folder, exist_ok=True)

    new_filename = f"{predicted_label}_{filename}"

    move(image_path, os.path.join(output_class_folder, new_filename))

    print(f"Processed {filename}")

with ThreadPoolExecutor() as executor:
    executor.map(process_image, [filename for filename in os.listdir(input_folder_path) if filename.endswith(".jpg") or filename.endswith(".png")])

print("Image sorting and renaming complete.")
