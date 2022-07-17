import pickle
import torch
from Model.Model import BC_stack
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
data_concat = []
subgoal = '0'
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
new_x_import = []
new_y_import = []
for traj in data:
    for i in range(len(traj["obs_robot"])-1):
        if i < (len(traj["obs_robot"])-1)*0.7:
            new_x.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]])
            new_y.append(traj["obs_robot"][i+1])
        else:
            new_x_import.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]])
            new_y_import.append(traj["obs_robot"][i+1])
np_x = np.array(new_x)
np_y = np.array(new_y)

np_x_imp = np.array(new_x_import)
np_y_imp = np.array(new_y_import)

np.save('./data_IGL/np_x_sg'+subgoal+'_no_imp',np_x)
np.save('./data_IGL/np_y_sg'+subgoal+'_no_imp',np_y)
np.save('./data_IGL/np_x_sg'+subgoal+'_imp',np_x_imp)
np.save('./data_IGL/np_y_sg'+subgoal+'_imp',np_y_imp)