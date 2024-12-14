import os
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ----------------------------
# Configuration and Setup
# ----------------------------

np.random.seed(42)

# Path to the ImageNet training directory
imagenet_train_folder = '/home/maria/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'

# Paths to save/load embeddings and labels
embeddings_path = 'embeddings_vit.npy'
labels_path = 'labels_vit.npy'
selected_classes_path = 'selected_classes_vit.npy'

# Choose whether to use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# ----------------------------
# Function Definitions
# ----------------------------

def load_embeddings(embeddings_path, labels_path, selected_classes_path):
    """
    Load embeddings, labels, and selected classes from disk.
    """
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    selected_classes = np.load(selected_classes_path, allow_pickle=True).tolist()
    return embeddings, labels, selected_classes

def save_embeddings(embeddings, labels, selected_classes, embeddings_path, labels_path, selected_classes_path):
    """
    Save embeddings, labels, and selected classes to disk.
    """
    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    np.save(selected_classes_path, selected_classes)
    print(f"Embeddings saved to {embeddings_path}")
    print(f"Labels saved to {labels_path}")
    print(f"Selected classes saved to {selected_classes_path}")

def get_image_files(folder_path):
    """
    Retrieve a list of image file paths from a given folder.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)]

def embed_images(image_paths, processor, model, device):
    """
    Embed a list of images using the provided ViT processor and model.
    """
    embeddings = []
    for idx, img_path in enumerate(image_paths):
        try:
            # Open and preprocess the image
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            # Get embeddings from the model
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.pooler_output.squeeze().cpu().numpy()
                embeddings.append(cls_embedding)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.array(embeddings)

# ----------------------------
# Main Script
# ----------------------------

def main():
    # Check if embeddings already exist
    if os.path.exists(embeddings_path) and os.path.exists(labels_path) and os.path.exists(selected_classes_path):
        print("Embeddings found. Loading embeddings, labels, and selected classes...")
        embeddings, labels, selected_classes = load_embeddings(embeddings_path, labels_path, selected_classes_path)
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        print(f"Selected classes: {selected_classes}")
    else:
        print("Embeddings not found. Processing images to generate embeddings...")
        
        # List all class folders
        class_folders = [d for d in os.listdir(imagenet_train_folder) 
                        if os.path.isdir(os.path.join(imagenet_train_folder, d))]
        print(f"Total classes found: {len(class_folders)}")
        
        # Ensure there are enough classes to choose from
        if len(class_folders) < 2:
            raise ValueError("Not enough class folders to select two randomly.")
        
        # Select two random classes without replacement
        two_random_folders = np.random.choice(class_folders, 2, replace=False)
        selected_classes = two_random_folders.tolist()
        print(f"Two randomly selected classes: {selected_classes}")
        
        # Initialize the processor and model
        model_name = 'google/vit-base-patch16-224'
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name).to(device)
        model.eval()  # Set model to evaluation mode
        print(f"Loaded ViT model '{model_name}'")
        
        # Initialize lists to hold embeddings and labels
        embeddings_list = []
        labels_list = []
        
        # Iterate over the two selected classes
        for label, class_name in enumerate(selected_classes):
            class_folder = os.path.join(imagenet_train_folder, class_name)
            image_files = get_image_files(class_folder)
            print(f"Processing class '{class_name}' with {len(image_files)} images.")
            
            # Embed images and collect embeddings
            class_embeddings = embed_images(image_files, processor, model, device)
            embeddings_list.append(class_embeddings)
            labels_list.extend([label] * len(class_embeddings))
        
        # Concatenate embeddings from both classes
        embeddings = np.vstack(embeddings_list)
        labels = np.array(labels_list)
        print(f"Total embeddings shape: {embeddings.shape}")
        print(f"Total labels shape: {labels.shape}")
        
        # Save embeddings and labels for future use
        save_embeddings(embeddings, labels, selected_classes, embeddings_path, labels_path, selected_classes_path)
    
    # ----------------------------
    # t-SNE Embedding
    # ----------------------------
    
    print("Starting t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_embedded = tsne.fit_transform(embeddings)
    print("t-SNE embedding completed.")
    
    # ----------------------------
    # Visualization
    # ----------------------------
    
    # Define colors for the two classes
    colors = ['red', 'blue']
    class_labels = selected_classes
    
    plt.figure(figsize=(10, 8))
    
    for class_idx in range(2):
        idxs = labels == class_idx
        plt.scatter(
            embeddings_embedded[idxs, 0],
            embeddings_embedded[idxs, 1],
            c=colors[class_idx],
            label=class_labels[class_idx],
            alpha=0.6,
            edgecolors='w',
            linewidth=0.5
        )
    
    plt.title('t-SNE Embedding of Images from Two ImageNet Classes using ViT')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
