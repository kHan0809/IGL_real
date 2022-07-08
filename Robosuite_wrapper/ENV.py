import numpy as np

class ENV(object):
    def __init__(self, env,stack_num):
        self.env = env
        self.stack_num = stack_num
        self.stacked_state      = 0
        self.stacked_next_state = 0

        self.robot_state_name_list, self.robot_state_dim  = self.state_func(self.env.observation_spec(),'robot')
        self.object_state_name_list,self.object_state_dim = self.state_func(self.env.observation_spec(),'object')

        self.one_state_dim = self.robot_state_dim + self.object_state_dim
        self.state_dim = (self.robot_state_dim + self.object_state_dim)*self.stack_num
        self.state_name_list = self.robot_state_name_list + self.object_state_name_list

        self.action_low_limit, self.action_high_limit = env.action_spec
        self.action_dim = self.action_low_limit.shape[0]


        if sum(abs(self.action_low_limit) == abs(self.action_high_limit)) != self.action_low_limit.shape[0]:
            print("Becareful : The action limit bound is different")

    def reset(self):
        obs = self.env.reset()
        # ===========state stack===================
        one_state = obs[self.state_name_list[0]]
        for state_ in self.state_name_list[1:]:
            if isinstance(obs[state_], np.float64):
                one_state = np.concatenate((one_state,np.array([obs[state_]])))
            else:
                one_state = np.concatenate((one_state, obs[state_]))

        stacked_state = one_state.copy()
        for i in range(self.stack_num-1):
            stacked_state = np.concatenate((stacked_state,one_state))

        self.stacked_state = stacked_state.copy()
        return stacked_state

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)

        one_next_state = next_obs[self.state_name_list[0]]
        for state_ in self.state_name_list[1:]:
            if isinstance(next_obs[state_], np.float64):
                one_next_state = np.concatenate((one_next_state, np.array([next_obs[state_]])))
            else:
                one_next_state = np.concatenate((one_next_state,next_obs[state_]))

        self.stacked_next_state = np.concatenate((self.stacked_state[self.one_state_dim:], one_next_state))
        self.stacked_state = self.stacked_next_state.copy()

        return self.stacked_state, reward, done, None

    @property
    def _max_episode_steps(self):
        return 500

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


    def state_func(self,observation_spec, state_name):
        name_list = []
        state_dim = 0
        if state_name == 'robot':
            for name in observation_spec.keys():
                if (('eef' in name) and ('robot' in name)) or (('gripper_qpos' in name) and ('robot' in name)):
                    name_list.append(name)
                    state_dim += observation_spec[name].shape[0]
        if state_name == 'object':
            for name in observation_spec.keys():
                if (('eef' not in name) and ('robot' not in name) and ('object' not in name)):
                    name_list.append(name)
                    if isinstance(observation_spec[name],int):
                        state_dim += 1
                    else:
                        state_dim += observation_spec[name].shape[0]
        return name_list, state_dim
