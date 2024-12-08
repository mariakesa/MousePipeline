import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm  # For progress bars
import os
import pickle

# 1. Load Embeddings
def load_embeddings(embeddings_path):
    """
    Load embeddings from a .npy file.
    
    Parameters:
    -----------
    embeddings_path : str
        Path to the embeddings .npy file.
    
    Returns:
    --------
    np.ndarray
        Embeddings matrix of shape (num_images, embedding_dim).
    """
    embeddings = np.load(embeddings_path).astype(np.float32)  # Use float32 for memory efficiency
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

# 2. Load .mat Data
def load_mat_data(mat_path):
    """
    Load event and sequence data from a .mat file.
    
    Parameters:
    -----------
    mat_path : str
        Path to the .mat file.
    
    Returns:
    --------
    np.ndarray
        Events matrix of shape (num_neurons, num_images).
    np.ndarray
        Sequences array of shape (num_images,).
    """
    mat = scipy.io.loadmat(mat_path)
    events = mat['stim'][0][0][1].T  # Transpose to shape (num_neurons, num_images)
    sequences = mat['stim'][0][0][2].flatten() - 1  # Adjust indexing if necessary
    print(f"Loaded events with shape: {events.shape}")
    print(f"Loaded sequences with shape: {sequences.shape}")
    return events, sequences

# 3. Filter Events
def filter_events(events, sequences, exclude_sequence=2800):
    """
    Filter out events corresponding to a specific sequence.
    
    Parameters:
    -----------
    events : np.ndarray
        Events matrix of shape (num_neurons, num_images).
    sequences : np.ndarray
        Sequences array of shape (num_images,).
    exclude_sequence : int, optional
        Sequence value to exclude. Default is 2800.
    
    Returns:
    --------
    np.ndarray
        Filtered events matrix of shape (num_neurons, num_selected_images).
    np.ndarray
        Filtered sequences array of shape (num_selected_images,).
    """
    mask = sequences != exclude_sequence
    filtered_events = events[:, mask]
    filtered_sequences = sequences[mask]
    print(f"Filtered events shape: {filtered_events.shape}")
    print(f"Filtered sequences shape: {filtered_sequences.shape}")
    return filtered_events, filtered_sequences

# 4. Compute STA for a Single Neuron
def compute_sta_single_neuron(event_times, embeddings_shuffled):
    """
    Compute the STA for a single neuron.
    
    Parameters:
    -----------
    event_times : np.ndarray
        Array of image indices where events occurred for the neuron.
    embeddings_shuffled : np.ndarray
        Shuffled embeddings matrix of shape (num_images, embedding_dim).
    
    Returns:
    --------
    np.ndarray
        STA vector of shape (embedding_dim,).
    """
    if event_times.size > 0:
        selected_embeddings = embeddings_shuffled[event_times, :]  # Shape: (num_events, embedding_dim)
        sta = selected_embeddings.mean(axis=0)
    else:
        sta = np.full(embeddings_shuffled.shape[1], np.nan, dtype=np.float32)
    return sta

# 5. Compute STA for All Neurons
def compute_sta_all_neurons(embeddings, events, sequences):
    """
    Compute the STA for all neurons.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings matrix of shape (num_images, embedding_dim).
    events : np.ndarray
        Events matrix of shape (num_neurons, num_images).
    sequences : np.ndarray
        Sequences array of shape (num_images,).
    
    Returns:
    --------
    np.ndarray
        STA matrix of shape (num_neurons, embedding_dim).
    """
    stas = []
    num_neurons = events.shape[0]
    print("Computing actual STA...")
    for neuron in tqdm(range(num_neurons), desc="Neurons"):
        event_times = np.where(events[neuron, :] != 0)[0]
        sta = compute_sta_single_neuron(event_times, embeddings)
        stas.append(sta)
    stas = np.array(stas, dtype=np.float32)
    print(f"Computed STA matrix with shape: {stas.shape}")
    return stas

# 6. Compute STA for Null Permutations
def compute_sta_null_permutation(embeddings, events, sequences, seed):
    """
    Compute the STA for a null permutation.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Original embeddings matrix of shape (num_images, embedding_dim).
    events : np.ndarray
        Events matrix of shape (num_neurons, num_images).
    sequences : np.ndarray
        Sequences array of shape (num_images,).
    seed : int
        Random seed for permutation.
    
    Returns:
    --------
    np.ndarray
        STA matrix of shape (num_neurons, embedding_dim).
    """
    np.random.seed(seed)
    embeddings_shuffled = np.random.permutation(embeddings)  # Shuffle rows (images)
    stas = []
    num_neurons = events.shape[0]
    for neuron in range(num_neurons):
        event_times = np.where(events[neuron, :] != 0)[0]
        sta = compute_sta_single_neuron(event_times, embeddings_shuffled)
        stas.append(sta)
    stas = np.array(stas, dtype=np.float32)
    return stas

# 7. Perform Permutation Test
def perform_permutation_test(embeddings, events, sequences, num_permutations=100, save_dir='permutation_results'):
    """
    Perform multiple STA permutations and perform a permutation test.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Original embeddings matrix of shape (num_images, embedding_dim).
    events : np.ndarray
        Events matrix of shape (num_neurons, num_images).
    sequences : np.ndarray
        Sequences array of shape (num_images,).
    num_permutations : int, optional
        Number of permutation iterations. Default is 100.
    save_dir : str, optional
        Directory to save permutation eigenvalues. Default is 'permutation_results'.
    
    Returns:
    --------
    np.ndarray
        Null eigenvalues matrix of shape (num_permutations, num_components).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    null_eigenvalues = []
    
    print(f"Performing {num_permutations} permutations...")
    for perm in tqdm(range(num_permutations), desc="Permutations"):
        seed = perm  # Ensures unique seed for each permutation
        stas_null = compute_sta_null_permutation(embeddings, events, sequences, seed)
        
        # Perform PCA on null STA
        pca_null = PCA()
        pca_null.fit(stas_null)
        eigenvalues_null = pca_null.explained_variance_
        null_eigenvalues.append(eigenvalues_null)
        
        # Save eigenvalues
        save_path = os.path.join(save_dir, f'eigenvalues_null_perm_{perm}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(eigenvalues_null, f)
    
    null_eigenvalues = np.array(null_eigenvalues)  # Shape: (num_permutations, num_components)
    print(f"Completed {num_permutations} permutations.")
    return null_eigenvalues

# 8. Permutation Test Analysis
def permutation_test_analysis(actual_eigenvalues, null_eigenvalues, alpha=0.05):
    """
    Analyze permutation test results to determine significance of actual eigenvalues.
    
    Parameters:
    -----------
    actual_eigenvalues : np.ndarray
        Eigenvalues from actual STA PCA. Shape: (num_components,).
    null_eigenvalues : np.ndarray
        Eigenvalues from null PCA permutations. Shape: (num_permutations, num_components).
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns:
    --------
    np.ndarray
        p-values for each principal component. Shape: (num_components,).
    np.ndarray
        Boolean array indicating significant components. Shape: (num_components,).
    """
    num_components = actual_eigenvalues.shape[0]
    p_values = np.zeros(num_components)
    significance = np.zeros(num_components, dtype=bool)
    
    for i in range(num_components):
        # Proportion of null eigenvalues >= actual eigenvalue
        p_val = np.mean(null_eigenvalues[:, i] >= actual_eigenvalues[i])
        p_values[i] = p_val
        significance[i] = p_val < alpha
    
    return p_values, significance

# 9. Main Execution Function
def main():
    # Define paths
    embeddings_path = '/home/maria/Documents/CarsenMariusData/6845348/embeddings.npy'
    mat_path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
    num_permutations = 100
    permutation_save_dir = "/home/maria/Documents/CarsenMariusData/6845348/permutation_results/"
    
    # Load data
    embeddings = load_embeddings(embeddings_path)
    events, sequences = load_mat_data(mat_path)
    
    # Filter events
    events_filtered, sequences_filtered = filter_events(events, sequences, exclude_sequence=2800)
    
    # Verify embeddings and events alignment
    if embeddings.shape[0] != sequences_filtered.shape[0]:
        raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) does not match number of sequences ({sequences_filtered.shape[0]}).")
    
    # Compute actual STA
    stas_actual = compute_sta_all_neurons(embeddings, events_filtered, sequences_filtered)
    
    # Perform PCA on actual STA
    print("Performing PCA on actual STA...")
    pca_actual = PCA()
    pca_actual.fit(stas_actual)
    actual_eigenvalues = pca_actual.explained_variance_
    actual_variance_ratio = pca_actual.explained_variance_ratio_
    cumulative_variance_actual = np.cumsum(actual_variance_ratio)
    
    print("Actual PCA completed.")
    print("Cumulative Variance Explained (Actual):")
    print(cumulative_variance_actual)
    
    # Perform permutations and collect null eigenvalues
    null_eigenvalues = perform_permutation_test(
        embeddings, 
        events_filtered, 
        sequences_filtered, 
        num_permutations=num_permutations, 
        save_dir=permutation_save_dir
    )
    
    # Perform PCA on null distributions is already handled inside perform_permutation_test
    
    # Permutation Test Analysis
    print("Analyzing permutation test results...")
    p_values, significance = permutation_test_analysis(actual_eigenvalues, null_eigenvalues, alpha=0.05)
    
    # Display Results
    for i, (eigen, p_val, sig) in enumerate(zip(actual_eigenvalues, p_values, significance), start=1):
        status = "Significant" if sig else "Not Significant"
        print(f"PC{i}: Eigenvalue = {eigen:.4f}, p-value = {p_val:.4f} --> {status}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_variance_actual)+1), cumulative_variance_actual, label='Actual PCA', marker='o')
    
    # Compute mean and std of null cumulative variances
    null_cumulative = np.cumsum(null_eigenvalues, axis=1)
    mean_null_cumulative = np.mean(null_cumulative, axis=0)
    std_null_cumulative = np.std(null_cumulative, axis=0)
    
    plt.fill_between(
        np.arange(1, len(mean_null_cumulative)+1),
        mean_null_cumulative - std_null_cumulative,
        mean_null_cumulative + std_null_cumulative,
        color='gray',
        alpha=0.3,
        label='Null Mean ± STD'
    )
    
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained: Actual vs. Null')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Save permutation test results
    results = {
        'actual_eigenvalues': actual_eigenvalues,
        'actual_variance_ratio': actual_variance_ratio,
        'cumulative_variance_actual': cumulative_variance_actual,
        'null_eigenvalues': null_eigenvalues,
        'p_values': p_values,
        'significance': significance
    }

    print(results)
    
    with open(os.path.join(permutation_save_dir, 'permutation_test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Permutation test results saved to {permutation_save_dir}")

# 10. Run the Main Function
if __name__ == "__main__":
    main()
