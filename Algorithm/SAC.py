import torch
import torch.nn.functional as F
import numpy as np
from Common.Buffer import ReplayBuffer
from Common.Utils import soft_update, hard_update
from torch.optim import Adam
from Model.Model import MLPQFunction_double, Squashed_Gaussian_Actor

class SAC(object):
    def __init__(self, state_dim, action_dim, args):

        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.policy_type = args.policy
        self.automatic_entropy_tuning = args.automatic_entropy_tuning


        self.critic      = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.buffer = ReplayBuffer(obs_dim=state_dim, act_dim=action_dim, size=self.buffer_size)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.Tensor([action_dim]).to(self.device).item()
                self.log_alpha = np.log([args.alpha]).astype(np.float32)
                self.log_alpha = torch.tensor(self.log_alpha, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = Squashed_Gaussian_Actor(state_dim, action_dim,args.hidden_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            raise Exception('No implementation')

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _ = self.policy(state)
        else:
            action, _ = self.policy(state,Eval=True)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        data = self.buffer.sample_batch(batch_size=batch_size)

        self.critic_optimizer.zero_grad()
        critic_loss, Q1, Q2 = self.compute_loss_q(data)
        # Optimize the critic
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        self.policy_optim.zero_grad()
        policy_loss = self.compute_loss_pi(data)
        policy_loss.backward()
        self.policy_optim.step()

        for p in self.critic.parameters():
            p.requires_grad = True


        if self.automatic_entropy_tuning:
            alpha_loss = self.compute_loss_alpha(data)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        with torch.no_grad():
            soft_update(self.critic_target, self.critic, self.tau)

        return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    def compute_loss_q(self,data):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = data['obs'], data['act'], data['rew'], data['obs2'], data['mask']

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        Q1, Q2 = self.critic(state_batch, action_batch)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action, next_log_pi = self.policy(next_state_batch)
            # Compute critic loss
            target_Q1, target_Q2 = self.critic_target(next_state_batch,next_action)
            minq = torch.min(target_Q1, target_Q2)
            target_y = reward_batch + self.gamma*mask_batch*(minq-self.alpha*next_log_pi)


        critic_loss = F.mse_loss(Q1,target_y) + F.mse_loss(Q2,target_y)

        return critic_loss, Q1, Q2
    def compute_loss_pi(self,data):
        state_batch = data['obs']
        state_batch = torch.FloatTensor(state_batch).to(self.device)

        pi, log_pi = self.policy(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * log_pi - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        return policy_loss
    def compute_loss_alpha(self,data):
        state_batch = data['obs']
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        pi, log_pi = self.policy(state_batch)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def save_checkpoint(self,algorithm, env_name, path):
        path = path + '/'+ algorithm + '_' + env_name + '.pth'
        print('Saving model to {}'.format(path))
        torch.save({'critic': self.critic.state_dict(),
                    'actor': self.policy.state_dict()}, path)

    def load_checkpoint(self, algorithm, env_name, path):
        path = path + '/' + algorithm + '_' + env_name + '.pth'
        print('load model : {}'.format(path))

