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

# Base filenames for saving/loading
embeddings_prefix = 'embeddings_vit_layer_'
labels_path = 'labels_vit.npy'
selected_classes_path = 'selected_classes_vit.npy'

# Choose whether to use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# ----------------------------
# Function Definitions
# ----------------------------

def get_image_files(folder_path):
    """
    Retrieve a list of image file paths from a given folder.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(valid_extensions)]

def embed_images_all_layers(image_paths, processor, model, device):
    """
    Embed a list of images using the provided ViT processor and model, 
    returning CLS embeddings from all layers for each image.
    """
    if len(image_paths) == 0:
        # No images, return empty arrays
        # We'll handle concatenation later
        with torch.no_grad():
            outputs = model(**processor(images=Image.new('RGB', (224,224)), return_tensors="pt").to(device))
        num_layers = len(outputs.hidden_states)
        return [np.empty((0, model.config.hidden_size)) for _ in range(num_layers)]
    
    # Run one sample to determine number of layers
    sample_image = Image.open(image_paths[0]).convert('RGB')
    sample_inputs = processor(images=sample_image, return_tensors="pt").to(device)
    with torch.no_grad():
        sample_outputs = model(**sample_inputs)
    num_layers = len(sample_outputs.hidden_states)

    # Prepare lists to store embeddings for each layer
    layer_embeddings = [[] for _ in range(num_layers)]

    for idx, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract CLS token from each layer
            # outputs.hidden_states[i] has shape (batch_size=1, seq_len, hidden_size)
            for i in range(num_layers):
                cls_token = outputs.hidden_states[i][0,0,:].cpu().numpy()
                layer_embeddings[i].append(cls_token)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert lists to numpy arrays of shape (num_images, hidden_size)
    layer_embeddings = [np.array(layer_emb) for layer_emb in layer_embeddings]
    return layer_embeddings

def save_metadata(labels, selected_classes, labels_path, selected_classes_path):
    np.save(labels_path, labels)
    np.save(selected_classes_path, selected_classes)
    print(f"Labels saved to {labels_path}")
    print(f"Selected classes saved to {selected_classes_path}")

def load_metadata(labels_path, selected_classes_path):
    labels = np.load(labels_path)
    selected_classes = np.load(selected_classes_path, allow_pickle=True).tolist()
    return labels, selected_classes

def all_embeddings_exist(num_layers):
    """
    Check if embeddings for all layers already exist.
    """
    for i in range(num_layers):
        if not os.path.exists(f"{embeddings_prefix}{i}.npy"):
            return False
    return os.path.exists(labels_path) and os.path.exists(selected_classes_path)

# ----------------------------
# Main Script
# ----------------------------

def main():
    model_name = 'google/vit-base-patch16-224'
    # Enable output_hidden_states to get all layers
    model = ViTModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(model_name)

    num_layers = model.config.num_hidden_layers + 1  # embedding layer + N transformer layers
    print(f"Model has {num_layers} layers of hidden states (including embedding layer).")

    # Check if embeddings for all layers already exist
    if all_embeddings_exist(num_layers):
        print("All layer embeddings found. Loading embeddings and metadata...")
        # Load embeddings
        layer_embeddings = [np.load(f"{embeddings_prefix}{i}.npy") for i in range(num_layers)]
        labels, selected_classes = load_metadata(labels_path, selected_classes_path)
    else:
        print("Embeddings not found. Processing images to generate all-layer embeddings...")
        
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
        
        embeddings_per_class = []
        labels_list = []
        
        # Iterate over the two selected classes
        for label, class_name in enumerate(selected_classes):
            class_folder = os.path.join(imagenet_train_folder, class_name)
            image_files = get_image_files(class_folder)
            print(f"Processing class '{class_name}' with {len(image_files)} images.")
            
            # Embed images and collect embeddings for all layers
            layer_embeddings_class = embed_images_all_layers(image_files, processor, model, device)
            embeddings_per_class.append(layer_embeddings_class)
            labels_list.extend([label] * layer_embeddings_class[0].shape[0])
        
        # Concatenate embeddings from both classes for each layer
        layer_embeddings = []
        for i in range(num_layers):
            layer_i_emb = np.vstack([embeddings_per_class[0][i], embeddings_per_class[1][i]])
            layer_embeddings.append(layer_i_emb)
        
        labels = np.array(labels_list)
        
        # Save all layer embeddings
        for i in range(num_layers):
            np.save(f"{embeddings_prefix}{i}.npy", layer_embeddings[i])
            print(f"Layer {i} embeddings saved to {embeddings_prefix}{i}.npy")
        
        # Save labels and selected classes
        save_metadata(labels, selected_classes, labels_path, selected_classes_path)

    # ----------------------------
    # t-SNE Embedding and Visualization per Layer
    # ----------------------------
    
    # Define colors for the two classes
    colors = ['red', 'blue']
    class_labels = selected_classes

    for layer_idx, layer_emb in enumerate(layer_embeddings):
        # layer_emb shape: (total_images, hidden_size)
        num_points = layer_emb.shape[0]
        print(f"Layer {layer_idx} embeddings shape: {layer_emb.shape} - plotting {num_points} points.")

        if num_points < 2:
            print(f"Not enough points ({num_points}) to run t-SNE for layer {layer_idx}. Skipping.")
            continue

        print(f"Starting t-SNE embedding for layer {layer_idx}...")
        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        embeddings_embedded = tsne.fit_transform(layer_emb)
        print(f"t-SNE embedding for layer {layer_idx} completed.")
        
        plt.figure(figsize=(10, 8))
        for class_idx in range(2):
            idxs = (labels == class_idx)
            plt.scatter(
                embeddings_embedded[idxs, 0],
                embeddings_embedded[idxs, 1],
                c=colors[class_idx],
                label=class_labels[class_idx],
                alpha=0.6,
                edgecolors='w',
                linewidth=0.5
            )
        
        plt.title(f't-SNE Embedding (Layer {layer_idx}) of Images from Two ImageNet Classes')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
