import pickle
import torch
from Model.Model import BC_stack
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
data_concat = []
subgoal = '4'
for pickle_data in os.listdir(os.getcwd()+'/data_IGL'):
    if 'Inter_using_mid_sg'+subgoal in pickle_data:
        print(pickle_data)
        with open('./data_IGL/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

new_x = []
new_y = []

for traj in data:
    for i in range(len(traj["obs_robot"])-1):
        new_x.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]])
        new_y.append(traj["obs_robot"][i+1])
np_x = np.array(new_x)
np_y = np.array(new_y)



np.save('./data_IGL/np_x_sg'+subgoal,np_x)
np.save('./data_IGL/np_y_sg'+subgoal,np_y)
