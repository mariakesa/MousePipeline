from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os
import pickle
from allen_vit_pipeline.utils import make_container_dict


class Config:
    def __init__(self, session, stimulus):
        load_dotenv()
        self.transformer_embedding_path=os.environ.get('HGMS_TRANSFORMER_EMBEDDING_PATH')
        self.allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        self.save_path=os.environ.get('HGMS_STA_PATH')
        self.boc = BrainObservatoryCache(manifest_file=str(Path(self.allen_cache_path) / Path('brain_observatory_manifest.json')))
        self.sessin=session
        self.stimulus=stimulus
        


#class ComputeSTA:

class EIDRepository:
    def __init__(self, config):
        self.config = config

    def get_existing_containers(self):

