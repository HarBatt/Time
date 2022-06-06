## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# 1. import models
from models.timegan import timegan
from models.rgan import rgan
from models.wrgan_gp import wrgan_gp
# 2. Data loading
from data_loading import real_data_loading
# 3. Utils
from utils import Parameters

#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Data loading
data_path = "data/"
dataset = "energy"
path_real_data = "data/" + dataset + "_data.csv"

#parameters

params = Parameters()
params.dataset = dataset
params.data_path = "data/" + params.dataset + "_data.csv"
params.model_save_path = "saved_models/" + params.dataset
params.seq_len = 24
params.batch_size = 128
params.max_steps = 100
params.gamma = 1.0 #Paramter for TimeGAN
params.save_model = True
params.print_every = 750
params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params.save_synth_data = False

params.latent_dim = 10 #Latent dim for the generator paramter for wrgan and rgan
params.disc_extra_steps = 1 #Extra steps for the discriminator; 5 is the original value from the paper.
                            #1 for rgan and 3 for wrgan
params.gp_lambda = 5 #Lambda for the gradient penalty from the paper. paramter for wrgan


#preprocessing the data.

"""
Method: real_data_loading()
---------------------------------------------------------------------------------------------------------------------
    - Loads the data from the path.
    - Scales the data using a min-max scaler.
    - Slices the data into windows of size seq_len.
    - Shuffles the data randomly, and returns it.
"""
ori_data, (minimum, maximum) = real_data_loading(path_real_data, params.seq_len)

params.input_size = ori_data[0].shape[1]
params.hidden_size = 24
params.num_layers = 3

print('Preprocessing Complete!')
   
with open(data_path + params.dataset + '_real_data.npy', 'wb') as f:
    np.save(f, np.array(ori_data))

print("Saved real data!")


generated_data = wrgan_gp(ori_data, params)  

# # Renormalization
# generated_data = generated_data*maximum
# generated_data = generated_data + minimum 

with open(data_path + params.dataset + '_synthetic_data.npy', 'wb') as f:
    np.save(f, np.array(generated_data))