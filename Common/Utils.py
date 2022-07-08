import numpy as np
import gym
import random
import torch
import torch.nn as nn


def Eval(eval_env, agent, eval_num,logger,render=False):
    reward_history = []
    max_action = float(eval_env.action_space.high[0])
    for j in range(eval_num):
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        while not(d or (ep_len == eval_env._max_episode_steps)):
            if (j == eval_num-1)&(render):
                eval_env.render()
            # Take deterministic actions at test time
            action = agent.select_action(o, evaluate=True)
            o, r, d, _ = eval_env.step(action*max_action)
            ep_ret += r
            ep_len += 1

        reward_history.append(ep_ret)
        logger.add_result("Test_return", ep_ret,type="min_max")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    # torch.manual_seed(random_seed)
    # np.random.seed(random_seed)
    # random.seed(random_seed)

    return random_seed


def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def weight_init_Xavier(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def logger(algo_name,iter,log_start = None, log_flag = False,dir=None,total_step=None,result=None):
    if log_start == None:
        raise Exception('Please set the \'log_start\' to True or False.')
    if log_start:
        if log_flag:
            if dir == None:
                f = open("./log " + algo_name +'_'+ str(iter) + ".txt", 'w')
                f.close()
            else:
                f = open(dir + algo_name +'_'+ str(iter) + ".txt", 'w')
                f.close()
        else:
            pass
    else:
        if log_flag:
            if dir == None:
                f = open("./log " + algo_name + '_' + str(iter) + ".txt", 'a')
                f.write(str(total_step))
                for i in range(len(result)):
                    f.write(" ")
                    f.write(str(int(result[i])))
                f.write("\n")
                f.close()
            else:
                f = open(dir + algo_name + '_' + str(iter) + ".txt", 'a')
                f.write(str(total_step))
                for i in range(len(result)):
                    f.write(" ")
                    f.write(str(int(result[i])))
                f.write("\n")
                f.close()
        else:
            pass

class dict_logger():
    def __init__(self,algo_name,env_name,seed,save):
        self.algo_name = algo_name
        self.env_name = env_name
        self.save = save
        self.seed = seed
        self.current = dict()
        self.first   = True
        self.min_max = ["Min","Ave","Max"]
        self.print_space = 20
        if self.save:
            f = open("./log/" + self.algo_name + '_' + self.env_name + '_' + str(self.seed) + ".txt", 'w')
            f.close()

    def add_result(self,name,add_element,type="_"):
        if type == "min_max":
            name = name + "_"
        if name in self.current.keys():
            self.current[name].append(add_element)
        else:
            self.current[name] = [add_element]


    def write(self,epoch):
        if self.save:
            f = open("./log/" + self.algo_name + '_' + self.env_name + '_' + str(self.seed) + ".txt", 'a')
            if self.first:
                f.write("Epoch")
                f.write(" ")
                for key in self.current.keys():
                    if key[-1] == "_":
                        for txt in self.min_max:
                            f.write(txt+key[:-1])
                            f.write(" ")
                    else:
                        f.write(key)
                        f.write(" ")
                f.write("\n")
                self.first = False

            f.write(str(epoch))
            f.write(" ")
            for key in self.current.keys():
                if key[-1] == "_":
                    f.write(str(round(min(self.current[key]),1)))
                    f.write(" ")
                    f.write(str(round(sum(self.current[key])/len(self.current[key]),1)))
                    f.write(" ")
                    f.write(str(round(max(self.current[key]),1)))
                    f.write(" ")
                else:
                    f.write(str(round(sum(self.current[key])/len(self.current[key]),1)))
                    f.write(" ")
            f.write("\n")
            f.close()
            self.current = dict()
        else:
            self.current = dict()
            pass

    def print_(self, epoch,step_per_epoch):
        print("----------------------------------------")
        print("{0: <{1}}".format("Epoch", self.print_space) + " : " + str(int(epoch)))
        print("{0: <{1}}".format("total_numsteps", self.print_space) + " : " + str(int(epoch*step_per_epoch)))
        for key in self.current.keys():
            if key[-1] == "_":
                key_str = "{0: <{1}}".format(self.min_max[0]+key[:-1], self.print_space) + " : " + str(round(min(self.current[key]), 2))
                print(key_str)
                key_str = "{0: <{1}}".format(self.min_max[1]+key[:-1], self.print_space) + " : " + str(round(sum(self.current[key]) / len(self.current[key]), 2))
                print(key_str)
                key_str = "{0: <{1}}".format(self.min_max[2]+key[:-1], self.print_space) + " : " + str(round(max(self.current[key]), 2))
                print(key_str)

            else:
                key_str = "{0: <{1}}".format(key, self.print_space) + " : " + str(round(sum(self.current[key]) / len(self.current[key]), 2))
                print(key_str)
        print("----------------------------------------")


