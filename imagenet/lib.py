import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ----------------------------
# 1. Data Preparation
# ----------------------------

# Define the main folder containing ImageNet classes
folder = '/home/maria/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'

# List all class folders
class_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
print(f"Total classes found: {len(class_folders)}")

# Ensure there are enough classes to choose from
if len(class_folders) < 2:
    raise ValueError("Not enough class folders to select two randomly.")

np.random.seed(42)
# Select two random class folders without replacement
two_random_folders = np.random.choice(class_folders, 2, replace=False)
print(f"Two randomly selected folders: {two_random_folders}")

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor()           # Convert PIL Image to Tensor
])

# Initialize lists to hold processed images and their corresponding labels
two_im_classes = []
labels = []

# Iterate over the two selected folders
for label, im_folder in enumerate(two_random_folders):
    folder_path = os.path.join(folder, im_folder)
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"Processing folder: {im_folder} with {len(image_files)} images.")
    
    for file in image_files:
        file_path = os.path.join(folder_path, file)
        try:
            # Open the image using PIL and ensure it's in RGB format
            with Image.open(file_path) as img:
                img = img.convert('RGB')  # Convert grayscale images to RGB

                # Apply the transformations
                transformed_img = transform(img)

                # Flatten the tensor and convert to NumPy array
                flattened_image = transformed_img.flatten().numpy()

                # Append to the list
                two_im_classes.append(flattened_image)
                labels.append(label)

        except (IOError, ValueError) as e:
            print(f"Skipping file {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error with file {file_path}: {e}")

# Convert the lists to NumPy arrays
images = np.array(two_im_classes)
labels = np.array(labels)

print(f"Total processed images: {images.shape[0]}")
print(f"Each image is flattened to shape: {images.shape[1:]}")

# ----------------------------
# 2. t-SNE Embedding
# ----------------------------

print("Starting t-SNE embedding...")
tsne = TSNE(n_components=2, random_state=42, verbose=1, perplexity=10)
images_embedded = tsne.fit_transform(images)
print("t-SNE embedding completed.")

# ----------------------------
# 3. Visualization
# ----------------------------

# Define colors for the two classes
colors = ['red', 'blue']
class_labels = two_random_folders

plt.figure(figsize=(10, 8))

for class_idx in range(2):
    idxs = labels == class_idx
    plt.scatter(
        images_embedded[idxs, 0],
        images_embedded[idxs, 1],
        c=colors[class_idx],
        label=class_labels[class_idx],
        alpha=0.6,
        edgecolors='w',
        linewidth=0.5
    )

plt.title('t-SNE Embedding of Images from Two ImageNet Classes')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
