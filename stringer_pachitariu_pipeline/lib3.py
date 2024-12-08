import numpy as np

embeddings=np.load('/home/maria/Documents/CarsenMariusData/6845348/embeddings.npy')
path="/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.decomposition import PCA

mat = scipy.io.loadmat(path)


events=mat['stim'][0][0][1].T
sequences=mat['stim'][0][0][2].flatten()-1
images_nonempty=events[:,sequences!=2800]
sequences=sequences[sequences!=2800]


def null(embeddings, events, sequences):
    embeddings=np.random.permutation(embeddings)
    stas=[]
    for neuron in range(events.shape[0]):
        event_times=events[neuron, :].nonzero()
        images=sequences[event_times[0]]
        if event_times:
            selected_embeddings=embeddings[images, :]
            sta=selected_embeddings.mean(axis=0)
        stas.append(sta)
    print(np.array(stas).shape)
    return np.array(stas)

def actual(embeddings, events, sequences):
    #embeddings=np.random.permutation(embeddings)
    stas=[]
    for neuron in range(events.shape[0]):
        event_times=events[neuron, :].nonzero()
        images=sequences[event_times[0]]
        if event_times:
            selected_embeddings=embeddings[images, :]
            sta=selected_embeddings.mean(axis=0)
        stas.append(sta)
    print(np.array(stas).shape)
    return np.array(stas)

start=time.time()
distr=actual(embeddings, images_nonempty, sequences)
#null_distr=null(embeddings, images_nonempty, sequences)
end=time.time()
print(end-start)

# Apply PCA
pca = PCA()
pca.fit(distr)

# Get the eigenvalues (explained variance) and the explained variance ratios
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Compute cumulative variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)




import numpy as np
import pickle
import matplotlib.pyplot as plt

# ================================
# 1. Load Actual PCA Results
# ================================

# Assuming you have computed the actual PCA in your script and have the following variables:
# eigenvalues, explained_variance_ratio, cumulative_variance

# If you have saved the actual PCA results to a file, load them here.
# For example, if saved using pickle:
#actual_pca_path = "/home/maria/Documents/CarsenMariusData/actual_pca_results.pkl"

# Example structure

# ================================
# 2. Load Permutation Null Distribution
# ================================

null_distr_path = "/home/maria/Documents/CarsenMariusData/permutation_result/null_distr.pkl"

with open(null_distr_path, 'rb') as f:
    permutation_results = pickle.load(f)

# Extract eigenvalues from all permutations
null_eigenvalues = [entry['eigenvalues'] for entry in permutation_results]
null_eigenvalues = np.array(null_eigenvalues)  # Shape: (num_permutations, num_components)

print(f"Loaded null eigenvalues with shape: {null_eigenvalues.shape}")

# ================================
# 3. Compute P-Values for Each Principal Component
# ================================

num_permutations, num_components = null_eigenvalues.shape

# Initialize p-values array
p_values = np.zeros(num_components)

for i in range(num_components):
    # Number of null eigenvalues >= actual eigenvalue
    count = np.sum(null_eigenvalues[:, i] >= eigenvalues[i])
    # P-value with continuity correction
    p_val = (count + 1) / (num_permutations + 1)
    p_values[i] = p_val

# ================================
# 4. Determine Significance
# ================================

alpha = 0.05  # Significance level
significant = p_values < alpha

# ================================
# 5. Display Results
# ================================

print("\nPermutation Test Results:")
print("----------------------------")
for i in range(num_components):
    status = "Significant" if significant[i] else "Not Significant"
    print(f"PC{i+1}: Eigenvalue = {eigenvalues[i]:.4f}, p-value = {p_values[i]:.4f} --> {status}")

# ================================
# 6. Visualization
# ================================

# Plot cumulative variance
plt.figure(figsize=(10, 6))
components = np.arange(1, num_components + 1)

# Actual cumulative variance
plt.plot(components, cumulative_variance, label='Actual PCA', marker='o')

# Null cumulative variance statistics
null_cumulative = np.cumsum(null_eigenvalues, axis=1)  # Shape: (num_permutations, num_components)
mean_null_cumulative = np.mean(null_cumulative, axis=0)
std_null_cumulative = np.std(null_cumulative, axis=0)

# Plot null mean and standard deviation
plt.fill_between(
    components,
    mean_null_cumulative - std_null_cumulative,
    mean_null_cumulative + std_null_cumulative,
    color='gray',
    alpha=0.3,
    label='Null Mean ± STD'
)

# Highlight significant PCs
for i in range(num_components):
    if significant[i]:
        plt.axvline(x=i+1, color='red', linestyle='--', alpha=0.5)

plt.xlabel('Principal Component')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained: Actual vs. Null Permutations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# 7. Save Analysis Results (Optional)
# ================================

analysis_results = {
    'eigenvalues': eigenvalues,
    'explained_variance_ratio': explained_variance_ratio,
    'cumulative_variance': cumulative_variance,
    'null_eigenvalues': null_eigenvalues,
    'p_values': p_values,
    'significant': significant
}

save_analysis_path = "/home/maria/Documents/CarsenMariusData/permutation_result/permutation_test_analysis.pkl"

with open(save_analysis_path, 'wb') as f:
    pickle.dump(analysis_results, f)

print(f"\nPermutation test analysis results saved to {save_analysis_path}")

    