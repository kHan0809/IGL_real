import torch
import torch.nn as nn
from Common.Utils import weight_init, weight_init_Xavier
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np


def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


class GenerativeGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        self.hidden_dim = list(hidden_dim)
        self.hidden_dim[0] += self.epsilon_dim

        self.net = mlp([obs_dim+self.epsilon_dim] + self.hidden_dim + [act_dim], activation, nn.Tanh())
        self.apply(weight_init_Xavier)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        if noise == 'gaussian':
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        return pi_action


class Squashed_Gaussian_Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_dim,activation=nn.ReLU(),log_std_min=-20, log_std_max=2):
        super(Squashed_Gaussian_Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden_dim = list(hidden_dim)
        self.net = mlp([obs_dim] + self.hidden_dim, activation)
        self.mu_layer = nn.Linear(self.hidden_dim[-1], act_dim)
        self.log_std_layer = nn.Linear(self.hidden_dim[-1], act_dim)

    def forward(self,state,Eval=False):
        output = self.net(state)
        mean = self.mu_layer(output)
        log_std = self.log_std_layer(output)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        pi_distribution = Normal(mean, std)

        if Eval:
            log_pi = pi_distribution.log_prob(mean).sum(axis=-1)
            log_pi -= (2*(np.log(2) - mean - F.softplus(-2*mean))).sum(axis=1)
            tanh_mean = torch.tanh(mean)

            return tanh_mean, log_pi
        else:
            sample_action = pi_distribution.rsample()
            log_pi = pi_distribution.log_prob(sample_action).sum(axis=-1)
            log_pi -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1)
            tanh_sample = torch.tanh(sample_action)

            return tanh_sample, log_pi
    #============tanh version==============
    # def forward(self,state,Eval=False):
    #     output = self.net(state)
    #     mean, log_std = output.chunk(2, dim=-1)
    #     log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    #     std = log_std.exp()
    #
    #     pi_distribution = Normal(mean, std)
    #
    #     if Eval:
    #         log_pi = pi_distribution.log_prob(mean).sum(axis=-1)
    #         log_pi -= (2*(np.log(2) - mean - F.softplus(-2*mean))).sum(axis=1)
    #         tanh_mean = torch.tanh(mean)
    #
    #         return tanh_mean, log_pi
    #     else:
    #         sample_action = pi_distribution.rsample()
    #         log_pi = pi_distribution.log_prob(sample_action).sum(axis=-1)
    #         log_pi -= (2*(np.log(2) - sample_action - F.softplus(-2*sample_action))).sum(axis=1)
    #         tanh_sample = torch.tanh(sample_action)
    #
    #         return tanh_sample, log_pi

    def entropy(self, state):
        dist = self.dist(state)
        return dist.entropy()


class MLPQFunction_double(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, activation=nn.ReLU()):
        super().__init__()
        self.hidden_dim = list(hidden_dim)

        self.q1 = mlp([obs_dim + act_dim] + self.hidden_dim + [1], activation)
        self.q2 = mlp([obs_dim + act_dim] + self.hidden_dim + [1], activation)
        self.apply(weight_init_Xavier)

    def forward(self, obs, act):
        q1 = self.q1(torch.cat([obs, act], dim=-1))
        q2 = self.q2(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q1, -1), torch.squeeze(q2, -1) # Critical to ensure q has right shape.

    def Q1(self,obs, act):
        q1 = self.q1(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q1, -1)


class Actor(nn.Module):
    def __init__(self,obs_dim, act_dim, hidden_dim, activation=nn.ReLU()):
        super(Actor, self).__init__()
        self.hidden_dim = list(hidden_dim)

        self.net = mlp([obs_dim] + self.hidden_dim + [act_dim], activation, nn.Tanh())
        self.apply(weight_init_Xavier)

    def forward(self,obs):
        action = self.net(obs)
        return action





# class Discriminator(nn.Module):
#     def __init__(self, num_inputs, hidden_sizes=(256,256)):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
#         self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.fc3 = nn.Linear(hidden_sizes[1], 1)
#
#         self.apply(weight_init)
#
#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         # prob = torch.sigmoid(self.fc3(x))
#         return self.fc3(x)

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_sizes=(256,256)):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc4 = nn.Linear(hidden_sizes[1], 1)
        self.LR = nn.LeakyReLU(0.2, inplace=True)
        self.apply(weight_init)

    def forward(self, x):
        x = self.LR(self.fc1(x))
        x = self.LR(self.fc2(x))
        x = self.LR(self.fc3(x))
        # prob = torch.sigmoid(self.fc3(x))
        return self.fc4(x)

class Discriminator_wp(nn.Module):
    def __init__(self, num_inputs, hidden_sizes=(256,256)):
        super(Discriminator_wp, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[1])
        self.fc4 = nn.Linear(hidden_sizes[1], 1)
        self.LR = nn.LeakyReLU(0.2, inplace=True)
        self.apply(weight_init)

    def forward(self, x):
        x = self.LR(self.fc1(x))
        x = self.LR(self.fc2(x))
        x = self.LR(self.fc3(x))
        return torch.squeeze(self.fc4(x), -1)


class Kernel(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Kernel, self).__init__()
        self.device = device
        self.apply(weight_init)


        self.state_hat_nn = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, state_dim*4),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim)
        )

        self.reward_hat_nn = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, state_dim),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(state_dim, 1)
        )

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action, _format=True):
        if _format is True:
            state, action = self._format(state, action)
        _input = torch.cat([state, action], dim=1)
        next_state_hat = self.state_hat_nn(_input)
        reward_hat = self.reward_hat_nn(_input)

        return next_state_hat, reward_hat.squeeze()

class BC(nn.Module):
    def __init__(self, state_dim, stage_dim,action_dim, device):
        super(BC, self).__init__()
        self.device = device
        self.apply(weight_init)


        self.net = nn.Sequential(
            nn.Linear(state_dim+stage_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, state_dim*4),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(state_dim*4, action_dim)
        )

    def forward(self, state, stage):
        _input=torch.cat([state, stage],dim=1)
        output = self.net(_input)
        return output

    def forward_(self, state, stage):
        _input=torch.cat([state, stage])
        output = self.net(_input)
        return output

class BC_stack(nn.Module):
    def __init__(self, state_dim, stage_dim,action_dim, device):
        super(BC_stack, self).__init__()
        self.device = device
        self.apply(weight_init)


        self.net = nn.Sequential(
            nn.Linear(state_dim+stage_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, state_dim*4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(state_dim*4, action_dim)
        )
    def forward(self, state):
        output = self.net(state)
        return output

class IGL(nn.Module):
    def __init__(self, all_dim, robot_dim, device):
        super(IGL, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.net = nn.Sequential(
            nn.Linear(all_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, robot_dim*4),
            nn.BatchNorm1d(robot_dim*4),
            nn.ReLU(),
            nn.Linear(robot_dim*4, robot_dim)
        )

    def forward(self, state):
        # _input=torch.cat([state, stage],dim=1)
        output = self.net(state)
        return output

    # def forward_(self, state, stage):
    #     _input=torch.cat([state, stage])
    #     output = self.net(_input)
    #     return output

class IGL_large(nn.Module):
    def __init__(self, all_dim, robot_dim, device):
        super(IGL_large, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.net = nn.Sequential(
            nn.Linear(all_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, robot_dim)
        )

    def forward(self, state):
        # _input=torch.cat([state, stage],dim=1)
        output = self.net(state)
        return output

class IGL_large_sep(nn.Module):
    def __init__(self, all_dim, robot_dim, device):
        super(IGL_large_sep, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.net = nn.Sequential(
            nn.Linear(all_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.net_pos = nn.Sequential(
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 3)
        )
        self.net_quat = nn.Sequential(
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 4)
        )
        self.net_grip = nn.Sequential(
        nn.Linear(256, 2)
        )

    def forward(self, state):
        # _input=torch.cat([state, stage],dim=1)
        common = self.net(state)
        pos = self.net_pos(common)
        quaternion = self.net_quat(common)
        quaternion_norm = torch.norm(quaternion,dim=1).unsqueeze(1)
        quaternion = quaternion/quaternion_norm
        grip = self.net_grip(common)
        output = torch.concat((pos,quaternion,grip),1)
        return output

class InvDyn(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(InvDyn, self).__init__()
        self.device = device

        self.InvDyn_net = nn.Sequential(
            nn.Linear(state_dim*2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim*4),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(action_dim*4, action_dim),
            nn.Tanh()
        )
        self.Predict_net = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, state_dim*4),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(state_dim*4, state_dim)
        )
        self.apply(weight_init)

    def forward(self, state,next_state):
        _input=torch.cat([state, next_state],dim=1)
        output = self.InvDyn_net(_input)
        return output

    def forward_(self, state,next_state):
        _input=torch.cat([state, next_state],dim=0)
        output = self.InvDyn_net(_input)
        return output

    def predict(self, state,action):
        _input=torch.cat([state, action],dim=1)
        output = self.Predict_net(_input)
        return output

class InvDyn_add(nn.Module):
    def __init__(self, state_dim,next_state_dim, action_dim, device):
        super(InvDyn_add, self).__init__()
        self.device = device

        self.InvDyn_net = nn.Sequential(
            nn.Linear(state_dim+next_state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim*4),
            nn.BatchNorm1d(action_dim*4),
            nn.LeakyReLU(),
            nn.Linear(action_dim*4, action_dim),
            nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, state,next_state):
        _input=torch.cat([state, next_state],dim=1)
        output = self.InvDyn_net(_input)
        return output

    def forward_(self, state,next_state):
        _input=torch.cat([state, next_state],dim=0)
        output = self.InvDyn_net(_input)
        return output

class InvDyn_consis_add(nn.Module):
    def __init__(self, state_dim,next_state_dim, action_dim, device):
        super(InvDyn_add, self).__init__()
        self.device = device

        self.InvDyn_net = nn.Sequential(
            nn.Linear(state_dim+next_state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim*4),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Linear(action_dim*4, action_dim),
            nn.Tanh()
        )

        self.Predict_net = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(256, next_state_dim*4),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(next_state_dim*4, next_state_dim)
        )
        self.apply(weight_init)

    def forward(self, state,next_state):
        _input=torch.cat([state, next_state],dim=1)
        output = self.InvDyn_net(_input)
        return output

    def forward_(self, state,next_state):
        _input=torch.cat([state, next_state],dim=0)
        output = self.InvDyn_net(_input)
        return output

    def predict(self, state,action):
        _input=torch.cat([state, action],dim=1)
        output = self.Predict_net(_input)
        return output