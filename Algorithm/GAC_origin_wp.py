import torch
import torch.nn.functional as F
import numpy as np
from Common.Buffer import ReplayBuffer
from Common.Utils import soft_update, hard_update
from torch.optim import Adam, RMSprop
from Model.Model import MLPQFunction_double, GenerativeGaussianMLPActor, Discriminator_wp
import torch.autograd as autograd
import itertools
import os
import time

class WGAC(object):
    def __init__(self, state_dim, action_dim, args):
        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.gamma = args.gamma
        self.tau = args.tau


        self.critic = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(device=self.device)
        self.q_optimizer = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(device=self.device)
        hard_update(self.critic_target, self.critic)


        self.buffer = ReplayBuffer(obs_dim=state_dim, act_dim=action_dim, size=self.buffer_size)
        self.obs_std = torch.FloatTensor(self.buffer.obs_std).to(device=self.device)
        self.obs_mean = torch.FloatTensor(self.buffer.obs_mean).to(device=self.device)

        # self.log_alpha = np.log([args.alpha]).astype(np.float32)
        # self.log_alpha = torch.tensor(self.log_alpha, requires_grad=True, device=self.device)
        # self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GenerativeGaussianMLPActor(state_dim, action_dim,args.hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.policy_target = GenerativeGaussianMLPActor(state_dim, action_dim,args.hidden_dim).to(self.device)
        hard_update(self.policy_target, self.policy)

        self.discrim = Discriminator_wp(state_dim + action_dim, args.hidden_dim).to(self.device)
        self.discrim_optim = RMSprop(self.discrim.parameters(), lr=args.lr*3)
        self.lambda_gp = args.lambda_gp



        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.entropy = args.entropy
        self.discount = args.discount


    def select_action(self, state, evaluate=False,state_limit=5.0):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate is False:
                action = self.policy_target(state)
            else:
                action = self.policy_target(state,std=0.5)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size,j):
        # Sample a batch from memory

        batch = self.buffer.sample_batch(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['mask']

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        # =========for entropy===============
        with torch.no_grad():
            uniform_action_ = (2*torch.rand_like(action_batch)-1)

            ##===================WP로 uniform 비슷하================================
            pol = self.discrim(torch.cat([state_batch.clone(), action_batch.clone()], dim=1))
            uni = self.discrim(torch.cat([state_batch.clone(), uniform_action_],dim=1))
            wp      = uni-pol
            wp      = wp.clamp(0,10)

        with torch.no_grad():
            next_state_action = self.policy_target(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target-wp*self.entropy)

        #===critic update===
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        #====================

        #===reducing computation===
        for p in self.critic.parameters():
            p.requires_grad = False
        # ==========================

        #=================Actor update======================
        state_batch_repeat = state_batch.clone()
        pi_repeat = self.policy(state_batch_repeat)

        qf1_pi, qf2_pi = self.critic(state_batch_repeat, pi_repeat)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)


        with torch.no_grad():
            uniform_action = (2*torch.rand_like(pi_repeat)-1)

        ##===================WP로 uniform 비슷하================================
        learner = self.discrim(torch.cat([state_batch_repeat, pi_repeat], dim=1))
        uniform = self.discrim(torch.cat([state_batch_repeat, uniform_action],dim=1))

        gradient_penalty = self.compute_gradient_penalty(torch.cat([state_batch_repeat, uniform_action],dim=1).cpu().detach().numpy(),
                                                         torch.cat([state_batch_repeat, pi_repeat], dim=1).cpu().detach().numpy())

        discrim_loss = - torch.mean(uniform) + torch.mean(learner) + self.lambda_gp * gradient_penalty
        # if j == 0:
        #     temp_state=state_batch_repeat[0].repeat(1000,1)
        #     temp_action = self.policy(temp_state)
        #
        #     with torch.no_grad():
        #         temp_uniform = (2 * torch.rand_like(temp_action) - 1)
        #
        #     temp_learner = self.discrim(torch.cat([temp_state, temp_action], dim=1))
        #     temp_uniform = self.discrim(torch.cat([temp_state, temp_uniform], dim=1))
        #
        #     print("==========================================")
        #
        #     print(torch.mean(temp_uniform)-torch.mean(temp_learner))

        self.discrim_optim.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optim.step()

        #=========================loss 다시 계산=======================================
        learner_ = self.discrim(torch.cat([state_batch_repeat.clone(), pi_repeat.clone()], dim=1))
        D_loss = - torch.mean(learner_)

        loss_pi = -min_qf_pi.mean() + D_loss*self.entropy

        self.policy_optim.zero_grad()
        loss_pi.backward()
        self.policy_optim.step()
        # ===========================================================================

        #===restore computation===
        for p in self.critic.parameters():
            p.requires_grad = True
        # ==========================

        with torch.no_grad():
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

        return qf1.mean().item(), qf2.mean().item(), qf_loss.mean().item(),  loss_pi.item(), D_loss.item(), self.entropy

    def entropy_change(self):
        self.entropy = self.entropy * self.discount


    def stable_baseline(self):
        self.obs_std = torch.FloatTensor(self.buffer.obs_std).to(device=self.device)
        self.obs_mean = torch.FloatTensor(self.buffer.obs_mean).to(device=self.device)

    def compute_gradient_penalty(self,expert_data,generate_data):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = np.random.random((expert_data.shape[0], 1))
        # Get random interpolation between real and fake samples
        interpolates = torch.FloatTensor((alpha * expert_data + ((1 - alpha) * generate_data))).requires_grad_(True).to(self.device)
        d_interpolates = self.discrim(interpolates)
        fake = torch.ones((expert_data.shape[0])).requires_grad_(False).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2 * result)
        grad_input[result == 0] = 0
        return grad_input