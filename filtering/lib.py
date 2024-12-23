from vit_pipeline.utils import make_container_dict
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import pickle

load_dotenv()

allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
boc = BrainObservatoryCache(manifest_file=str(Path(allen_cache_path) / Path('brain_observatory_manifest.json')))

experiment_containers = make_container_dict(boc)

session_A=experiment_containers[643061996]['three_session_A']
data_set_events= boc.get_ophys_experiment_events(session_A)

dat = boc.get_ophys_experiment_data(session_A)
stim_table = dat.get_stimulus_table('natural_movie_one')

transformer_embedding_path = Path("/home/maria/Documents/HuggingMouseData/TransformerEmbeddings/google_vit-base-patch16-224_embeddings.pkl") 
        # Load the transformer embeddings
with open(transformer_embedding_path, 'rb') as file:
    transfr = pickle.load(file)
embedding = transfr['natural_movie_one']  # Shape: (total_time_points, embedding_dim)
embedding_dim = embedding.shape[1]

first_trial=stim_table[0:900]

all_cells=data_set_events[:,first_trial['start']]

def get_neuron_trial_embeddings(neuron_index, all_cells, embedding, stim_table, num_trials=10, trial_length=900):
    """
    Compute the trial-wise embeddings for a given neuron.
    
    Args:
        neuron_index (int): Index of the neuron to analyze.
        all_cells (np.ndarray): Neural event data (num_cells x total_time_points).
        embedding (np.ndarray): Transformer embeddings (total_time_points x embedding_dim).
        stim_table (pd.DataFrame): Stimulus table containing 'start' times for trials.
        num_trials (int): Number of trials to process.
        trial_length (int): Number of time points per trial (default: 900).
        
    Returns:
        np.ndarray: A (num_trials x embedding_dim) array where each row corresponds to
                    the filtered embedding for a trial for the specified neuron.
    """
    embedding_dim = embedding.shape[1]
    trial_embeddings = np.zeros((num_trials, embedding_dim))
    
    # Verify that stim_table has enough trials
    if len(stim_table) < num_trials:
        print(f"Warning: Requested {num_trials} trials, but stim_table has only {len(stim_table)} trials.")
        num_trials = len(stim_table)
    
    print(f"Processing {num_trials} trials...")
    
    for trial_idx in range(10):
        try:
            # Access the start time of the trial using .iloc
            trial_start = stim_table.iloc[trial_idx*900:900*(trial_idx+1)]['start']
            trial_end = trial_start + trial_length  # Each trial has 900 time points
            #print(trial_start)

            # Extract the neural activity for this neuron during the trial
            neuron_activity = data_set_events[:,trial_start] # Shape: (900,)
            #print(f"Neuron activity shape: {neuron_activity.shape}")
            
            # Extract the corresponding embeddings for the trial
            trial_embedding = embedding  # Shape: (900, embedding_dim)
            #print(f"Trial embedding shape: {trial_embedding.shape}")

            v=neuron_activity[neuron_index].T
            X=trial_embedding

            filtered=v.T@X

            print(filtered.shape)
            # Assign the filtered embedding to the trial_embeddings array
            trial_embeddings[trial_idx, :] = filtered
        except IndexError as e:
            print(f"Error processing trial {trial_idx + 1}: {e}")
            trial_embeddings[trial_idx, :] = np.zeros(embedding_dim)
        except KeyError as e:
            print(f"KeyError accessing 'start' for trial {trial_idx + 1}: {e}")
            trial_embeddings[trial_idx, :] = np.zeros(embedding_dim)
    
    return trial_embeddings

# Example usage:
neuron_index = 0  # Specify the neuron index (adjust as needed)
num_trials = 10   # Specify the number of trials to process
trial_length = 900  # Number of time points per trial

# Compute trial-wise embeddings for the specified neuron
neuron_trial_embeddings = get_neuron_trial_embeddings(
    neuron_index=neuron_index,
    all_cells=all_cells,
    embedding=embedding,
    stim_table=stim_table,
    num_trials=num_trials,
    trial_length=trial_length
)

print(f"Neuron {neuron_index} Trial Embeddings Shape: {neuron_trial_embeddings.shape}")
