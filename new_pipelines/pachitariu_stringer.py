from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA
import scipy.io
import time

load_dotenv()

class Config:
    '''
    natimg2800_M160825_MP027_2016-12-14.mat
    natimg2800_M161025_MP030_2017-05-29.mat
    natimg2800_M170604_MP031_2017-06-28.mat
    natimg2800_M170714_MP032_2017-08-07.mat
    natimg2800_M170714_MP032_2017-09-14.mat
    natimg2800_M170717_MP033_2017-08-20.mat
    natimg2800_M170717_MP034_2017-09-11.mat
    '''
    def __init__(self, experiment_file, transformer_name='google/vit-base-patch16-224'):
        self.data_path = os.environ.get('SP_DATA_PATH')
        self.experiment_file=experiment_file
        self.experiment_path=Path(self.data_path)/Path(self.experiment_file+'.mat')
        self.transformer_name = transformer_name
        self.transformer_embedding_path = os.environ.get('SP_TRANSF_EMBEDDING_PATH')
        self.transformer_file=self.transformer_name.replace('/','_')+ '.npy'
        self.embedding_path=Path(self.transformer_embedding_path)/Path(self.transformer_file)
        self.save_path=os.environ.get('SP_ANALYSIS_PATH')


class FilterData:
    def __init__(self, config):
        """
        Initializes the DataSTA object by loading data, applying sparsification,
        and preparing sequences and embeddings.
        
        Parameters:
        - config (object): Configuration object containing necessary paths and parameters.
        """
        self.config = config
        
        # Load MATLAB .mat file
        self.mat = scipy.io.loadmat(config.experiment_path)
        
        # Extract and sparsify events
        self.events = self.mat['stim'][0][0][1].T  # Shape: (num_neurons, num_timepoints)
        
        # Extract and adjust sequences
        self.sequences = self.mat['stim'][0][0][2].flatten() - 1  # Adjusting indices if necessary
        
        # Filter out sequences with value 2800
        valid_mask = self.sequences != 2800
        self.events = self.events[:, valid_mask]        # Now shape: (num_neurons, num_valid_timepoints)
        self.sequences = self.sequences[valid_mask]     # Now shape: (num_valid_timepoints,)
        
        # Load embeddings
        self.embeddings = np.load(config.embedding_path)
        #plt.imshow(self.embeddings)
        #plt.show()
        
        # Debugging output
        print('Initialization complete.')
        print(f'Events shape: {self.events.shape}')
        print(f'Sequences shape after filtering: {self.sequences.shape}')
        print(f'Embeddings shape: {self.embeddings.shape}')

    def compute_filter(self):
        """
        Computes the Spike-Triggered Average (STA) for each neuron.
        
        Returns:
        - np.ndarray: Array of STAs with shape (num_neurons, embedding_dimension).
        """
        #plt.plot(self.events[0, :])
        #plt.show()

        images=self.embeddings[self.sequences]
        print(images.shape), print(self.events.shape)
        filtered=images.T@self.events.T

        return filtered
        
    def run(self):
        """
        Executes the STA computation and PCA analysis.
        
        Returns:
        - tuple: (pca_object, eigenvalues, explained_variance_ratio, cumulative_variance)
        """
        # Compute STAs
        filtered = self.compute_filter()
        
        return filtered

    
config=Config('natimg2800_M161025_MP030_2017-05-29')
filtered=FilterData(config).run()
#plt.imshow(np.corrcoef(pcs[0].T))
#plt.show()
print(filtered.shape)
print(np.corrcoef(filtered.T).shape)
plt.hist(np.corrcoef(filtered.T).flatten(), bins=100)
plt.show()

