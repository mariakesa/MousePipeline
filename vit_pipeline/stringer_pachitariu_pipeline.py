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
    def __init__(self, transformer_name='google/vit-base-patch16-224'):
        self.transformer_name = transformer_name
        self.transformer_embedding_path = os.environ.get('SP_TRANSF_EMBEDDING_PATH')
        self.transformer_file=self.transformer_name.replace('/','_')+ '.pkl'
        self.embedding_path=Path(self.transformer_embedding_path)/Path(self.transformer_file)
        self.save_path=os.environ.get('SP_ANALYSIS_PATH')