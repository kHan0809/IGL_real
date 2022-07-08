import pickle
import torch
from Model.Model import BC_stack
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
with open('./data_IGL/Inter_traj_using_middle.pickle', 'rb') as f:
    data = pickle.load(f)
new_x = []
new_y = []
new_x_import = []
new_y_import = []
for traj in data:
    for i in range(len(traj["obs_robot"])-1):
        if i < (len(traj["obs_robot"])-1)*0.9:
            new_x.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]])
            new_y.append(traj["obs_robot"][i+1])
        else:
            new_x_import.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]])
            new_y_import.append(traj["obs_robot"][i+1])
np_x = np.array(new_x)
np_y = np.array(new_y)

np_x_imp = np.array(new_x_import)
np_y_imp = np.array(new_y_import)

np.save('./data_IGL/np_x_no_imp',np_x)
np.save('./data_IGL/np_y_no_imp',np_y)
np.save('./data_IGL/np_x_imp',np_x_imp)
np.save('./data_IGL/np_y_imp',np_y_imp)