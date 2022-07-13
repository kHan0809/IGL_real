import pickle
import torch
from Model.Model import IGL, IGL_large
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class CustomDataSet(Dataset):
    def __init__(self,numpy_x_name,numpy_y_name,dir):
        self.x = np.load(dir+numpy_x_name)
        self.y = np.load(dir+numpy_y_name)
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
dataset1 = CustomDataSet('np_x_sg2_no_imp.npy','np_y_sg2_no_imp.npy','./data_IGL/')
dataset2 = CustomDataSet('np_x_sg2_imp.npy','np_y_sg2_imp.npy','./data_IGL/')

grid_lr    = [0.0001, 0.00005, 0.00001]
grid_wd    = [0,1e-4,1e-5,1e-6]
grid_batch = [100,50,25]


for x,batch in enumerate(grid_batch):
    train_loader1 = DataLoader(dataset1, shuffle = True,batch_size = 1000)
    train_loader2 = DataLoader(dataset2, shuffle = True,batch_size = batch)

    epochs = 120
    all_dim = 24
    robot_dim = 9
    device = "cuda"
    agent=IGL_large(all_dim,robot_dim,device)
    agent.to(device)
    # print(agent)
    # optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001,weight_decay=1e-5)
    for y,lr in enumerate(grid_lr):
        for z, wd in enumerate(grid_wd):
            optimizer = torch.optim.Adam(agent.parameters(), lr=lr,weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

            agent.train()
            loss = nn.MSELoss()
            for i in range(epochs):
                temp_loss1 = 0
                temp_loss2 = 0
                for k,(state,label) in enumerate(train_loader1):
                    optimizer.zero_grad()
                    output=agent(state.type(torch.FloatTensor).to(device))
                    loss_ = loss(label.type(torch.FloatTensor).to(device),output)
                    loss_.backward()
                    optimizer.step()
                    temp_loss1 += loss_.item()
                for j in range(2):
                    for k, (state, label) in enumerate(train_loader2):
                        optimizer.zero_grad()
                        output = agent(state.type(torch.FloatTensor).to(device))
                        loss_ = loss(label.type(torch.FloatTensor).to(device), output)
                        loss_.backward()
                        optimizer.step()
                        temp_loss2 += loss_.item()
                scheduler.step()
                print("========",i,"========")
                print(temp_loss1,temp_loss2)
            torch.save(agent.state_dict(), './model_save/IGL_sg2_imp'+str(x)+str(y)+str(z))