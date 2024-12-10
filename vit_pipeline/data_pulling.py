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

for c in experiment_containers.keys():
    session_A=experiment_containers[c]['three_session_A']
    #data_set_regression = boc.get_ophys_experiment_data(session_A)
    data_set_events= boc.get_ophys_experiment_events(session_A)