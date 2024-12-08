import numpy as np
import scipy.io
import time
from sklearn.decomposition import PCA
import pickle
import os
from concurrent.futures import ProcessPoolExecutor

# Load data
embeddings = np.load('embeddings.npy')
path = "/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
mat = scipy.io.loadmat(path)

events = mat['stim'][0][0][1].T
sequences = mat['stim'][0][0][2].flatten() - 1
images_nonempty = events[:, sequences != 2800]
sequences = sequences[sequences != 2800]

seeds = range(100)

def null(embeddings, events, sequences, seeds, index):
    print(index)
    seed = seeds[index]
    np.random.seed(seed)
    embeddings = np.random.permutation(embeddings)
    stas = []
    for neuron in range(events.shape[0]):
        event_times = events[neuron, :].nonzero()
        images = sequences[event_times[0]]
        if event_times:
            selected_embeddings = embeddings[images, :]
            sta = selected_embeddings.mean(axis=0)
        stas.append(sta)
    # Apply PCA
    pca = PCA()
    pca.fit(stas)

    # Get the eigenvalues (explained variance) and the explained variance ratios
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute cumulative variance explained
    cumulative_variance_null = np.cumsum(explained_variance_ratio)

    dat = {'eigenvalues': eigenvalues, 'explained_variance_ratio': explained_variance_ratio, 'cumulative_variance': cumulative_variance_null}
    return dat

# Parallel processing
start = time.time()
dct_lst = []

# Define a wrapper function for parallel execution
def parallel_null(index):
    return null(embeddings, images_nonempty, sequences, seeds, index)

with ProcessPoolExecutor() as executor:
    dct_lst = list(executor.map(parallel_null, range(100)))

end = time.time()
print(f"Time taken: {end - start} seconds")

# Save results
file_path = "/home/maria/Documents/CarsenMariusData/permutation_result/null_distr.pkl"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as file:
    pickle.dump(dct_lst, file)
