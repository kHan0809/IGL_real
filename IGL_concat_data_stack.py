import pickle
import torch
from Model.Model import BC_stack
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


with open('./data_IGL/Inter_traj_using_mid.pickle', 'rb') as f:
    data = pickle.load(f)
new_x = []
new_y = []
new_x_import = []
new_y_import = []
for traj in data:
    new_x.append(traj["obs_robot"][0] + traj["obs_obj"][0] + [traj["sg"][0]]
                 + traj["obs_robot"][0] + traj["obs_obj"][0] + [traj["sg"][0]]
                 + traj["obs_robot"][0] + traj["obs_obj"][0] + [traj["sg"][0]])
    new_y.append(traj["obs_robot"][1])

    new_x.append(traj["obs_robot"][0] + traj["obs_obj"][0] + [traj["sg"][0]]
                 + traj["obs_robot"][0] + traj["obs_obj"][0] + [traj["sg"][0]]
                 + traj["obs_robot"][1] + traj["obs_obj"][1] + [traj["sg"][1]])
    new_y.append(traj["obs_robot"][2])
    for i in range(len(traj["obs_robot"])-3):

        if i < (len(traj["obs_robot"])-1)*0.9:
            new_x.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]]
                         + traj["obs_robot"][i+1] + traj["obs_obj"][i+1] + [traj["sg"][i+1]]
                         + traj["obs_robot"][i+2] + traj["obs_obj"][i+2] + [traj["sg"][i+2]])
            new_y.append(traj["obs_robot"][i+3])
        else:
            new_x_import.append(traj["obs_robot"][i] + traj["obs_obj"][i] + [traj["sg"][i]]
                         + traj["obs_robot"][i+1] + traj["obs_obj"][i+1] + [traj["sg"][i+1]]
                         + traj["obs_robot"][i+2] + traj["obs_obj"][i+2] + [traj["sg"][i+2]])
            new_y_import.append(traj["obs_robot"][i+3])


np_x = np.array(new_x)
np_y = np.array(new_y)

np_x_imp = np.array(new_x_import)
np_y_imp = np.array(new_y_import)
# print(np_x.shape)
# print(np_y.shape)
# print(np_x_imp.shape)
# print(np_y_imp.shape)
np.save('./data_IGL/np_x_stack_nimp',np_x)
np.save('./data_IGL/np_y_stack_nimp',np_y)
np.save('./data_IGL/np_x_stack_imp',np_x_imp)
np.save('./data_IGL/np_y_stack_imp',np_y_imp)

