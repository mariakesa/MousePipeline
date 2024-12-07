from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import pickle
import pandas as pd

load_dotenv()

class Config:
    def __init__(self, session, stimulus, transformer_name='google_vit-base-patch16-224-in21k_embeddings'):
        self.transformer_name = transformer_name
        self.transformer_embedding_path = os.environ.get('HGMS_TRANSF_EMBEDDING_PATH')
        self.allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        self.save_path=os.environ.get('HGMS_STA_PATH')
        self.boc = BrainObservatoryCache(manifest_file=str(Path(self.allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.session = session
        self.stimulus = stimulus
        

class EIDRepository:
    def __init__(self, config):
        self.config = config

    def make_container_dict(self):
        '''
        Parses which experimental id's (values)
        correspond to which experiment containers (keys).
        '''
        boc=self.config.boc
        experiment_container = boc.get_experiment_containers()
        container_ids = [dct['id'] for dct in experiment_container]
        eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
        df = pd.DataFrame(eids)
        reduced_df = df[['id', 'experiment_container_id', 'session_type']]
        grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])[
            'id'].agg(list).reset_index()
        eid_dict = {}
        for row in grouped_df.itertuples(index=False):
            container_id, session_type, ids = row
            if container_id not in eid_dict:
                eid_dict[container_id] = {}
            eid_dict[container_id][session_type] = ids[0]
        return eid_dict

    def get_downloaded_eids(self):
        path=str(Path(self.config.allen_cache_path) / Path('ophys_experiment_events'))
        filenames = os.listdir(path)
        eids=[]
        for f in filenames:
            parsed=f.split('_')
            eid=int(parsed[0])
            eids.append(eid)
        experiment_containers=self.make_container_dict()
        session=[experiment_containers[k][self.config.session] for k in experiment_containers.keys() if self.config.session in experiment_containers[k].keys()]
        print('Session:', session)  
        relevant_eids=[eid for eid in eids if eid in session]
        return relevant_eids
    
    def get_already_processed_eids(self):
        path=self.config.save_path
        filenames = os.listdir(path)
        processed_eids=[]
        for f in filenames:
            parsed=f.split('_')
            print(parsed)
            if parsed[1]=='STA.npy':
                eid=int(parsed[0])
                processed_eids.append(eid)
        return processed_eids
    
    def get_eids_to_process(self):
        downloaded_eids=self.get_downloaded_eids()
        processed_eids=self.get_already_processed_eids()
        print(processed_eids)
        eids_to_process=[eid for eid in downloaded_eids if eid not in processed_eids]
        return eids_to_process

class STAProcessEID:
    def __init__(self, config):
        """
        Initialize the STA processing class.

        Parameters:
        -----------
        config : object
            Configuration object containing paths and parameters.
            Expected attributes:
                - transformer_embedding_path: Path to the transformer embeddings.
                - transformer_name: Name of the transformer model (e.g., 'ViT').
                - stimulus: Name of the stimulus (e.g., 'movie_one').
                - save_path: Directory to save the STA results.
                - boc: BrainObservatoryCache instance for data access.
        """
        self.config = config
        transformer_embedding_path = Path(self.config.transformer_embedding_path) / f"{self.config.transformer_name}.pkl"
        
        # Load the transformer embeddings
        with open(transformer_embedding_path, 'rb') as file:
            transfr = pickle.load(file)
        self.embedding = transfr[self.config.stimulus]  # Shape: (total_time_points, embedding_dim)
        self.embedding_dim = self.embedding.shape[1]
    
    def compute_sta(self, events, n_neurons, embedding_trial):
        """
        Compute STA increments from events for a given trial.

        Parameters:
        -----------
        events : tuple of (neuron_indices, time_indices)
            Indices of neurons and their event times within the trial.
        n_neurons : int
            Total number of neurons.
        embedding_trial : np.ndarray
            Embedding vectors for the trial's time points. Shape: (trial_time_points, embedding_dim)

        Returns:
        --------
        sta_increment : np.ndarray
            Sum of embeddings at event times for each neuron. Shape: (n_neurons, embedding_dim)
        count_increment : np.ndarray
            Number of events per neuron. Shape: (n_neurons,)
        """
        neuron_indices, time_indices = events
        embedding_dim = self.embedding_dim
        
        # Initialize accumulators
        sta_increment = np.zeros((n_neurons, embedding_dim))
        count_increment = np.zeros(n_neurons, dtype=int)
        
        if len(neuron_indices) > 0:
            # Accumulate embeddings for each event
            sta_increment[neuron_indices] += embedding_trial[time_indices]
            # Count events per neuron
            counts = np.bincount(neuron_indices, minlength=n_neurons)
            count_increment += counts
        
        return sta_increment, count_increment
    
    def __call__(self, eid):
        """
        Process a single experiment to compute and save the STA.

        Parameters:
        -----------
        eid : int or str
            Experiment ID to process.

        Returns:
        --------
        sta_final : np.ndarray
            Final STA for each neuron. Shape: (n_neurons, embedding_dim)
        """
        # Access the Brain Observatory Cache
        boc = self.config.boc
        
        # Retrieve event data and regression data for the experiment
        data_set_events = boc.get_ophys_experiment_events(eid)  # Shape: (n_neurons, total_time_points)
        data_set_regression = boc.get_ophys_experiment_data(eid)
        stim_table = data_set_regression.get_stimulus_table(self.config.stimulus)
        
        # Determine the number of neurons and trials
        n_neurons = data_set_events.shape[0]
        n_trials = stim_table['repeat'].nunique()
        
        # Initialize accumulators for STA computation
        sta_sum = np.zeros((n_neurons, self.embedding_dim))
        count_sum = np.zeros(n_neurons, dtype=int)
        
        # Iterate over each trial to accumulate STA sums and counts
        for trial in stim_table['repeat'].unique():
            # Extract absolute time points for this trial
            ts = stim_table.loc[stim_table['repeat'] == trial, 'start'].values  # Shape: (trial_time_points,)
            
            # Subset events and embeddings for this trial
            trial_events = data_set_events[:, ts].nonzero()  # Tuple: (neuron_indices, time_indices)
            embedding_trial = self.embedding        # Shape: (trial_time_points, embedding_dim)
            
            # Compute STA increments for this trial
            sta_increment, count_increment = self.compute_sta(trial_events, n_neurons, embedding_trial)
            #print(sta_increment.shape, count_increment)
            
            # Accumulate the sums and counts
            sta_sum += sta_increment
            count_sum += count_increment
        
        # Compute the final STA by averaging the sums
        sta_final = np.zeros((n_neurons, self.embedding_dim))
        valid_neurons = count_sum > 0
        sta_final[valid_neurons] = sta_sum[valid_neurons] / count_sum[valid_neurons, None]

        #print(sta_final.shape)
        #print(sta_final)
        
        # Save the STA to disk
        save_path = Path(self.config.save_path) / f"{eid}_STA.npy"
        np.save(save_path, sta_final)
        print(f"STA saved to {save_path}")
        
        return sta_final
    
class Gather:
    def __init__(self, config):
        self.config = config

    def gather(self):
        processed_files = os.listdir(self.config.save_path)
        arrays = []
        
        for fname in processed_files:
            if fname.endswith('.npy'):
                file_path = os.path.join(self.config.save_path, fname)
                arr = np.load(file_path)
                arrays.append(arr)

        # Stack all arrays vertically into a single large array
        if arrays:
            mega_array = np.vstack(arrays)
            return mega_array
        else:
            return np.array([])  # Return an empty array if no npy files found

