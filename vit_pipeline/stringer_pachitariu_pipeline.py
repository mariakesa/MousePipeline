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
    def __init__(self, experiment_file, quantile_sparsity=None, transformer_name='google/vit-base-patch16-224'):
        self.data_path = os.environ.get('SP_DATA_PATH')
        self.experiment_file=experiment_file
        self.experiment_path=Path(self.data_path)/Path(self.experiment_file+'.mat')
        self.transformer_name = transformer_name
        self.transformer_embedding_path = os.environ.get('SP_TRANSF_EMBEDDING_PATH')
        self.transformer_file=self.transformer_name.replace('/','_')+ '.npy'
        self.embedding_path=Path(self.transformer_embedding_path)/Path(self.transformer_file)
        self.save_path=os.environ.get('SP_ANALYSIS_PATH')
        self.quantile_sparsity=quantile_sparsity


def process_sparsity(events, config):
    """
    Applies quantile-based sparsification to the event data.
    
    For each neuron (row), values above the specified quantile are retained,
    and values below are set to zero.
    
    Parameters:
    - events (np.ndarray): 2D array of event data with shape (num_neurons, num_timepoints).
    - config (object): Configuration object containing sparsity parameters.
        - config.quantile_sparsity (bool): Flag to apply sparsification.
        - config.quantile_value (float): Quantile value between 0 and 1 (e.g., 0.95 for 95th percentile).
    
    Returns:
    - np.ndarray: Sparsified event data with the same shape as input.
    """
    if not config.quantile_sparsity:
        return events
    
    # Validate quantile_value
    quantile = config.quantile_sparsity
    if not (0 < quantile < 1):
        raise ValueError("quantile_value must be between 0 and 1.")
    
    # Initialize a copy to avoid modifying the original data
    sparsified_events = np.zeros_like(events)
    
    # Compute quantiles for each neuron (row-wise)
    neuron_quantiles = np.quantile(events, quantile, axis=1, keepdims=True)
    
    # Apply threshold: retain values above the quantile, set others to zero
    sparsified_events = np.where(events >= neuron_quantiles, events, 0)
    
    return sparsified_events

class DataSTA:
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
        original_events = self.mat['stim'][0][0][1].T  # Shape: (num_neurons, num_timepoints)
        self.events = process_sparsity(original_events, config)
        
        # Extract and adjust sequences
        self.sequences = self.mat['stim'][0][0][2].flatten() - 1  # Adjusting indices if necessary
        
        # Filter out sequences with value 2800
        valid_mask = self.sequences != 2800
        self.events = self.events[:, valid_mask]        # Now shape: (num_neurons, num_valid_timepoints)
        self.sequences = self.sequences[valid_mask]     # Now shape: (num_valid_timepoints,)
        
        # Load embeddings
        self.embeddings = np.load(config.embedding_path)
        print(self.embeddings.shape)
        print(self.embeddings[0,0], self.embeddings[1,0])
        #plt.imshow(self.embeddings)
        #plt.show()
        
        # Debugging output
        print('Initialization complete.')
        print(f'Original events shape: {original_events.shape}')
        print(f'Sparsified events shape: {self.events.shape}')
        print(f'Sequences shape after filtering: {self.sequences.shape}')
        print(f'Embeddings shape: {self.embeddings.shape}')

    def compute_sta(self):
        """
        Computes the Spike-Triggered Average (STA) for each neuron.
        
        Returns:
        - np.ndarray: Array of STAs with shape (num_neurons, embedding_dimension).
        """
        #plt.plot(self.events[0, :])
        #plt.show()
        num_neurons, num_timepoints = self.events.shape
        embedding_dim = self.embeddings.shape[1]
        stas = np.zeros((num_neurons, embedding_dim))
        
        for neuron in range(num_neurons):
            # Find indices where events occurred (non-zero)
            event_times = np.nonzero(self.events[neuron, :])[0]
            
            if event_times.size > 0:
                # Retrieve corresponding image indices from sequences
                images = self.sequences[event_times]
                
                # Validate image indices to prevent IndexError
                if np.any(images >= self.embeddings.shape[0]) or np.any(images < 0):
                    raise IndexError(f"Image indices out of bounds for neuron {neuron}.")
                
                # Select embeddings corresponding to the images
                selected_embeddings = self.embeddings[images, :]
                
                # Compute the mean embedding (STA)
                stas[neuron, :] = selected_embeddings.mean(axis=0)
            else:
                # If no events, STA remains zero
                pass  # Already initialized to zero
        
        return stas

    def run(self):
        """
        Executes the STA computation and PCA analysis.
        
        Returns:
        - tuple: (pca_object, eigenvalues, explained_variance_ratio, cumulative_variance)
        """
        # Compute STAs
        sta = self.compute_sta()



        #plt.imshow(np.cov(sta))
        #plt.show()

        plt.plot(sta[:,0])
        plt.plot(sta[:,1])
        plt.show()
        
        # Perform PCA
        pca = PCA()
        pca.fit(sta)
        
        # Extract PCA results
        eigenvalues = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        return sta, pca, eigenvalues, explained_variance_ratio, cumulative_variance
    

import numpy as np
import scipy.io
from sklearn.decomposition import PCA

def process_sparsity(events, config):
    """
    Applies quantile-based sparsification to the event data.

    For each neuron (row), values above the specified quantile are retained,
    and values below are set to zero.

    Parameters:
    - events (np.ndarray): 2D array of event data with shape (num_neurons, num_timepoints).
    - config (object): Configuration object containing sparsity parameters.
        - config.quantile_sparsity (bool): Flag to apply sparsification.
        - config.quantile_value (float): Quantile value between 0 and 1 (e.g., 0.95 for 95th percentile).

    Returns:
    - np.ndarray: Sparsified event data with the same shape as input.
    """
    if not config.quantile_sparsity:
        return events

    # Validate quantile_value
    quantile = config.quantile_value
    if not (0 < quantile < 1):
        raise ValueError("quantile_value must be between 0 and 1.")

    # Compute quantiles for each neuron (row-wise)
    neuron_quantiles = np.quantile(events, quantile, axis=1, keepdims=True)

    # Apply threshold: retain values above the quantile, set others to zero
    sparsified_events = np.where(events >= neuron_quantiles, events, 0)

    return sparsified_events

class DataSTAWeighted:
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
        original_events = self.mat['stim'][0][0][1].T  # Shape: (num_neurons, num_timepoints)
        self.events = process_sparsity(original_events, config)

        # Extract and adjust sequences
        self.sequences = self.mat['stim'][0][0][2].flatten() - 1  # Adjusting indices if necessary

        # Filter out sequences with value 2800
        valid_mask = self.sequences != 2800
        self.events = self.events[:, valid_mask]        # Now shape: (num_neurons, num_valid_timepoints)
        self.sequences = self.sequences[valid_mask]     # Now shape: (num_valid_timepoints,)

        # Load embeddings
        self.embeddings = np.load(config.embedding_path)

        # Debugging output
        print('Initialization complete.')
        print(f'Original events shape: {original_events.shape}')
        print(f'Sparsified events shape: {self.events.shape}')
        print(f'Sequences shape after filtering: {self.sequences.shape}')
        print(f'Embeddings shape: {self.embeddings.shape}')

    def compute_sta(self):
        """
        Computes the Spike-Triggered Average (STA) for each neuron using weighted averaging
        based on the magnitude of deconvolved events.

        Returns:
        - np.ndarray: Array of STAs with shape (num_neurons, embedding_dimension).
        """
        num_neurons, num_timepoints = self.events.shape
        embedding_dim = self.embeddings.shape[1]
        stas = np.zeros((num_neurons, embedding_dim))

        for neuron in range(num_neurons):
            # Find indices where events occurred (non-zero)
            event_times = np.nonzero(self.events[neuron, :])[0]

            if event_times.size > 0:
                # Retrieve corresponding image indices from sequences
                images = self.sequences[event_times]

                # Retrieve event magnitudes
                event_magnitudes = self.events[neuron, event_times]  # Shape: (num_events,)

                # Validate image indices to prevent IndexError
                if np.any(images >= self.embeddings.shape[0]) or np.any(images < 0):
                    raise IndexError(f"Image indices out of bounds for neuron {neuron}.")

                # Select embeddings corresponding to the images
                selected_embeddings = self.embeddings[images, :]  # Shape: (num_events, embedding_dim)

                # Compute the weighted sum of embeddings
                weighted_sum = np.dot(event_magnitudes, selected_embeddings)  # Shape: (embedding_dim,)

                # Compute the sum of event magnitudes for normalization
                total_weight = np.sum(event_magnitudes)

                if total_weight > 0:
                    # Compute the weighted average (STA)
                    stas[neuron, :] = weighted_sum / total_weight
                else:
                    # If total_weight is zero, leave STA as zero
                    pass
            else:
                # If no events, STA remains zero
                pass  # Already initialized to zero

        return stas

    def run(self):
        """
        Executes the STA computation and PCA analysis.

        Returns:
        - tuple: (pca_object, eigenvalues, explained_variance_ratio, cumulative_variance)
        """
        # Compute STAs
        sta = self.compute_sta()

        # Perform PCA
        pca = PCA()
        pca.fit(sta)

        # Extract PCA results
        eigenvalues = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        return sta,pca, eigenvalues, explained_variance_ratio, cumulative_variance

    
#config=Config('natimg2800_M161025_MP030_2017-05-29', quantile_sparsity=0)
#pcs=DataSTA(config).run()
#print(pcs)

