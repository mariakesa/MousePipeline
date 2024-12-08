path='/home/maria/Documents/CarsenMariusData/6845348/images_natimg2800_all.mat'

import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat(path)
print(mat.keys())
print(mat['imgs'].shape)

#plt.imshow(mat['imgs'][:,90:180,0])
#plt.show()
from transformers import ViTImageProcessor, ViTModel
model='google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model)
model = ViTModel.from_pretrained(model)

import torch
import numpy as np
def process_stims(stims):
    def get_pooler_dim(single_stim, processor, model):
        inputs = processor(images=single_stim, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.pooler_output.squeeze().detach().numpy()
        return cls.shape[-1]
    import time
    start = time.time()
    n_stims = len(stims)
    # n_stims=10
    stims_dim = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
    single_stim = stims_dim[0]
    pooler_dim = get_pooler_dim(single_stim, processor, model)
    embeddings = np.empty((n_stims, pooler_dim))
    for i in range(n_stims):
        # print(i)
        inputs = processor(images=stims_dim[i], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.pooler_output.squeeze().detach().numpy()
        embeddings[i, :] = cls
    end = time.time()
    print('Time taken for embedding one movie: ', end-start)
    return embeddings

stims=np.array([mat['imgs'][:,90:180,i] for i in range(2800)])
embeddings=process_stims(stims)
print(embeddings.shape)
np.save('/home/maria/Documents/CarsenMariusData/6845348/embeddings.npy',embeddings)