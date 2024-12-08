import numpy as np

embeddings=np.load('/home/maria/Documents/CarsenMariusData/6845348/embeddings.npy')
path="/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.decomposition import PCA
import json

mat = scipy.io.loadmat(path)



events=mat['stim'][0][0][1].T
sequences=mat['stim'][0][0][2].flatten()-1
images_nonempty=events[:,sequences!=2800]
sequences=sequences[sequences!=2800]

seeds=range(100)


def null(embeddings, events, sequences, seeds, index):
    seed=seeds[index]
    np.random.seed(seed)
    embeddings=np.random.permutation(embeddings)
    stas=[]
    for neuron in range(events.shape[0]):
        event_times=events[neuron, :].nonzero()
        images=sequences[event_times[0]]
        if event_times:
            selected_embeddings=embeddings[images, :]
            sta=selected_embeddings.mean(axis=0)
        stas.append(sta)
    # Apply PCA
    pca = PCA()
    pca.fit(stas)

    # Get the eigenvalues (explained variance) and the explained variance ratios
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # Compute cumulative variance explained
    cumulative_variance_null = np.cumsum(explained_variance_ratio)

    dat={'eigenvalues':eigenvalues, 'explained_variance_ratio':explained_variance_ratio, 'cumulative_variance':cumulative_variance_null}


    return dat

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
    return np.array(stas)

start=time.time()
distr=actual(embeddings, images_nonempty, sequences)

dct_lst=[]
for i in range(100):
    print(i)
    dct=null(embeddings, images_nonempty, sequences, seeds, i)
    dct_lst.append(dct)
end=time.time()
print(end-start)

json.dump(dct_lst, open("/home/maria/Documents/CarsenMariusData/6845348/permutation_results/null_distr.json", 'w'))

'''
# Apply PCA
pca = PCA()
pca.fit(distr)

# Get the eigenvalues (explained variance) and the explained variance ratios
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Compute cumulative variance explained
cumulative_variance = np.cumsum(explained_variance_ratio)

#print("\nEigenvalue Spectrum (Explained Variance per Component):")
#print(eigenvalues)

#print("\nVariance Explained Ratio (Per Component):")
#print(explained_variance_ratio)

print("\nCumulative Variance Explained:")
print(cumulative_variance)

# Apply PCA
pca = PCA()
pca.fit(null_distr)

# Get the eigenvalues (explained variance) and the explained variance ratios
eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Compute cumulative variance explained
cumulative_variance_null = np.cumsum(explained_variance_ratio)

#print("\nEigenvalue Spectrum (Explained Variance per Component):")
#print(eigenvalues)

#print("\nVariance Explained Ratio (Per Component):")
#print(explained_variance_ratio)

print("\nCumulative Variance Explained:")
print(cumulative_variance_null)

import matplotlib.pyplot as plt

plt.plot(cumulative_variance, label='Actual')
plt.plot(cumulative_variance_null, label='Null')
plt.legend()
plt.show()
'''
    