from Model.Model import MLPQFunction_double, Actor
import torch
import torch.nn.functional as F
from Common.Utils import soft_update, hard_update
from Common.Buffer import ReplayBuffer

class TD3():

    def __init__(self, state_dim, action_dim, args):
        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu")


        self.actor      = Actor(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.actor_target = Actor(state_dim, action_dim, args.hidden_dim).to(self.device)
        hard_update(self.actor_target, self.actor)


        self.critic      = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = MLPQFunction_double(state_dim, action_dim, args.hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.buffer = ReplayBuffer(obs_dim=state_dim, act_dim=action_dim, size=self.buffer_size)

    def select_action(self, state, evaluate=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def update_parameters(self, batch_size, updates, logger):
        data = self.buffer.sample_batch(batch_size=batch_size)

        self.critic_optimizer.zero_grad()
        critic_loss, Q1, Q2 = self.compute_loss_q(data)
        logger.add_result("Q1", Q1.mean().item())
        logger.add_result("Q2", Q2.mean().item())
        logger.add_result("critic_loss", critic_loss.item())
        critic_loss.backward()
        self.critic_optimizer.step()

        if (updates%2)==0:
            for p in self.critic.parameters():
                p.requires_grad = False

            self.actor_optimizer.zero_grad()
            actor_loss = self.compute_loss_pi(data)
            logger.add_result("actor_loss", actor_loss.item())
            actor_loss.backward()
            self.actor_optimizer.step()

            for p in self.critic.parameters():
                p.requires_grad = True

            # Update tareget networks
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.actor_target,  self.actor, self.tau)

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
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state_batch) + noise).clamp(-1.,1.)

            # Compute critic loss
            target_Q1, target_Q2 = self.critic_target(next_state_batch,next_action)
            minq = torch.min(target_Q1, target_Q2)
            target_y = reward_batch + self.gamma*mask_batch*minq


        critic_loss = F.mse_loss(Q1,target_y) + F.mse_loss(Q2,target_y)

        return critic_loss, Q1, Q2

    def compute_loss_pi(self,data):
        state_batch = data['obs']
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        pi = self.actor(state_batch)
        actor_loss = -self.critic.Q1(state_batch, pi).mean()
        return actor_loss

    def save_checkpoint(self,algorithm, env_name, path):
        path = path + '/'+ algorithm + '_' + env_name + '.pth'
        print('Saving model to {}'.format(path))
        torch.save({'critic': self.critic.state_dict(),
                    'actor': self.actor.state_dict()}, path)

    def load_checkpoint(self, algorithm, env_name, path):
        path = path + '/' + algorithm + '_' + env_name + '.pth'
        print('load model : {}'.format(path))

        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


