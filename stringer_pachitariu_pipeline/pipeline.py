from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA

load_dotenv()

class Config:
    def __init__(self, session, stimulus, transformer_name='google/vit-base-patch16-224'):
        self.transformer_name = transformer_name
        self.transformer_embedding_path = os.environ.get('HGMS_TRANSF_EMBEDDING_PATH')
        self.save_path=os.environ.get('HGMS_STA_PATH')
        self.session = session
        self.stimulus = stimulus