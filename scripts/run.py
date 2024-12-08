from allen_vit_pipeline.pipeline import Config, EIDRepository, STAProcessEID, Gather
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


config = Config('three_session_A', 'natural_movie_one')

eids = EIDRepository(config).get_eids_to_process()
print(eids)
print(len(eids))

'''

processor=STAProcessEID(config)

import time
start=time.time()
for eid in eids:
    processor(eid)
end=time.time()
print(end-start)'''

import time

start=time.time()
mega_array=Gather(config).gather()
print(mega_array.shape)
end=time.time()
print(end-start)

# Apply PCA
pca = PCA()
pca.fit(mega_array)

# Get the eigenvalues (explained variance) and the explained variance ratios
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Compute cumulative variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nEigenvalue Spectrum (Explained Variance per Component):")
print(eigenvalues)

print("\nVariance Explained Ratio (Per Component):")
print(explained_variance_ratio)

print("\nCumulative Variance Explained:")
print(cumulative_variance)


# Assume explained_variance_ratio and cumulative_variance are already defined from PCA
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))

# Plot explained variance ratio
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    explained_variance_ratio,
    marker='o',
    linestyle='-',
    label='Explained Variance Ratio'
)

plt.show()
# Plot cumulative variance explained
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o',
    linestyle='--',
    label='Cumulative Variance Explained'
)

plt.title('PCA Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.legend()
plt.show()
