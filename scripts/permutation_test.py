from vit_pipeline.allen_pipeline import Config, EIDRepository, PermutationRepository, Gather, PermutationSTA
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

config = Config('three_session_A', 'natural_movie_one')

eids = PermutationRepository(config).get_eids_to_process()
print('boom eids:', eids)

permuter=PermutationSTA(config, 0)(eids)