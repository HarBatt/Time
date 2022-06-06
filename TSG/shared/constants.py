import torch

class Parameters(object):
  """
  class: 
    - Parameters: store all the parameters
  """
  def __init__(self):
    pass


data_path = "data/"
dataset = "energy"
path_real_data = "data/" + dataset + "_data.csv"
save_synth_data = True
#set the device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_path = "saved_models/" + dataset