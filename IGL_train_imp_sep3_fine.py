import pickle
import torch
from Model.Model import IGL, IGL_large_sep
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
subgoal = '3'
dataset1 = CustomDataSet('np_x_sg'+subgoal+'_no_imp.npy','np_y_sg'+subgoal+'_no_imp.npy','./data_IGL/')
dataset2 = CustomDataSet('np_x_sg'+subgoal+'_imp.npy','np_y_sg'+subgoal+'_imp.npy','./data_IGL/')

grid_lr    = [0.0001, 0.00005,0.00001]
grid_wd    = [1e-4,1e-5]
grid_batch = [250,100,50]

for x,batch in enumerate(grid_batch):
    train_loader1 = DataLoader(dataset1, shuffle = True,batch_size = batch)
    train_loader2 = DataLoader(dataset2, shuffle = True,batch_size = batch//2)

    # print(agent)
    # optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001,weight_decay=1e-5)
    for y,lr in enumerate(grid_lr):
        for z, wd in enumerate(grid_wd):
            epochs = 30
            all_dim = 24
            robot_dim = 9
            device = "cuda"
            agent = IGL_large_sep(all_dim, robot_dim, device)
            agent.load_state_dict(torch.load('./model_save/BEST/SEP_IGL_sg3_imp212'))
            agent.to(device)
            optimizer = torch.optim.Adam(agent.parameters(), lr=lr,weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.9)

            agent.train()
            loss = nn.MSELoss()
            for i in range(epochs):
                temp_loss1 = 0
                temp_loss2 = 0
                for k,(state,label) in enumerate(train_loader1):
                    optimizer.zero_grad()
                    output=agent(state.type(torch.FloatTensor).to(device))
                    loss_ = loss(label.type(torch.FloatTensor).to(device) , output)
                    if i != 0:
                        loss_.backward()
                        optimizer.step()
                    temp_loss1 += loss_.item()
                for j in range(2):
                    for k, (state, label) in enumerate(train_loader2):
                        optimizer.zero_grad()
                        output = agent(state.type(torch.FloatTensor).to(device))
                        loss_ = loss(label.type(torch.FloatTensor).to(device), output)
                        if i != 0:
                            loss_.backward()
                            optimizer.step()
                        temp_loss2 += loss_.item()
                if i != 0:
                    scheduler.step()
                print("========",i,"========")
                print(temp_loss1,temp_loss2)
            torch.save(agent.state_dict(), './model_save/SEP_IGL_sg'+subgoal+'_imp_fine'+str(x)+str(y)+str(z))