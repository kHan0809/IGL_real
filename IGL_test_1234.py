import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from Model.Model import IGL, InvDyn_add, IGL_large
import torch

def get_current_stage(one_state):
    flag = 0
    inner = 0.021
    inner_pro = 1.1
    # print("----aaaa----")
    # print(one_state[:3])
    # print(one_state[9:12])
    # print(abs(one_state[:3]-one_state[9:12]))
    if abs(one_state[0] - one_state[9]) < inner and abs(one_state[1] - one_state[10]) < inner and abs(one_state[2] - one_state[11]) < inner:
        flag = 1
    if abs(one_state[0] - one_state[9]) < inner*inner_pro and abs(one_state[1] - one_state[10]) < inner*inner_pro and abs(one_state[2] - one_state[11]) < inner*inner_pro and abs(one_state[7]<0.026):
        flag = 2
    if abs(one_state[0] - one_state[9]) < inner*inner_pro and abs(one_state[1] - one_state[10]) < inner*inner_pro and abs(one_state[2] - one_state[11]) < inner*inner_pro and abs(one_state[7]<0.026) and abs(one_state[11] - one_state[-5])>0.035:
        flag = 3
    return flag


def _flatten_obs(obs_dict, name_list):
    ob_lst = []
    for key in name_list:
        if key in obs_dict:
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst)

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    # options["env_name"] = "Door"
    # options["env_name"] = choose_environment()
    options["env_name"] = "Stack"

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = "Panda"

    # Choose controller
    # controller_name = choose_controller()
    controller_name = "OSC_POSE"
    # controller_name = "OSC_POSITION"
    # controller_name = "IK_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    print(options["controller_configs"])
    print(type(options["controller_configs"]))
    # Help message to user
    print('Press "H" to show the viewer control panel.')

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    obs = env.reset()

    # obs_robot_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    # obs_obj_list  = ["cubeA_pos", "cubeA_quat","cubeB_pos","cubeB_quat"]

    obs_robot_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos","robot0_joint_pos_cos","robot0_joint_pos_sin","robot0_joint_vel","robot0_gripper_qvel"]
    obs_obj_list  = ["cubeA_pos", "cubeA_quat","cubeB_pos","cubeB_quat"]
    obs_robot_pos_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    temp_list = ["robot0_eef_pos"]

    obs_robot = _flatten_obs(obs,obs_robot_list)
    obs_obj = _flatten_obs(obs,obs_obj_list)

    obs_robot_pos = _flatten_obs(obs,obs_robot_pos_list)

    one_state = np.concatenate((obs_robot_pos, obs_obj))
    current_subgoal = np.array([get_current_stage(one_state)])
    # current_subgoal = np.array([0])

    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    all_dim = 24 # 9 + 13 + 1
    robot_dim = 9
    igl0 = IGL_large(all_dim, robot_dim, 'cpu')
    igl0.load_state_dict(torch.load('./model_save/IGL_sg_imp210'))
    igl1 = IGL_large(all_dim, robot_dim, 'cpu')
    igl1.load_state_dict(torch.load('./model_save/IGL_sg1_imp123'))
    igl2 = IGL_large(all_dim, robot_dim, 'cpu')
    igl2.load_state_dict(torch.load('./model_save/IGL_sg2_imp012'))
    igl3 = IGL_large(all_dim, robot_dim, 'cpu')
    igl3.load_state_dict(torch.load('./model_save/IGL_sg3_imp202'))

    state_dim = 32
    next_state_dim = 9
    action_dim = 7

    Inv = InvDyn_add(state_dim,next_state_dim, action_dim, 'cpu')
    # Inv.load_state_dict(torch.load('./model_save/InvDyn_4.pth'))


    igl0.eval()
    igl1.eval()
    igl2.eval()
    igl3.eval()
    Inv.eval()
    from collections import deque
    from scipy.spatial.transform import Rotation as R
    key = " "
    while True:
        abnormal_count = 0
        abnormal_buffer = deque(maxlen=5)
        abnormal_buffer.append(0),abnormal_buffer.append(1),abnormal_buffer.append(1),abnormal_buffer.append(1),abnormal_buffer.append(1)

        maintain = deque(maxlen=3)
        maintain.append(0),maintain.append(0),maintain.append(0)
        count = 0
        for i in range(500):
            # if maintain[0] != maintain[1] or maintain[0] != maintain[2] or maintain[2] != maintain[1]:
            #     print("-----------")
            #     current_subgoal = maintain[0]
            one_state = np.concatenate((one_state, current_subgoal))
            if current_subgoal == 0:
                next_= igl0(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 1:
                next_ = igl1(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 2:
                next_ = igl2(torch.FloatTensor(one_state).unsqueeze(0))
            else:
                next_ = igl3(torch.FloatTensor(one_state).unsqueeze(0))


            # action=Inv.forward(torch.FloatTensor(obs_robot).unsqueeze(0),next_)
            next = next_.squeeze(0).detach().numpy()
            action_pos = np.array([(next[0]-obs_robot_pos[0]),(next[1]-obs_robot_pos[1]),(next[2]-obs_robot_pos[2])])*10
            # if current_subgoal == 2:
            #     action_pos[0] = 0.0
            #     action_pos[1] = 0.0
            #     action_pos[2] = 0.5
            # action_pos = (obs_obj[:3] - obs_robot_pos[:3])


            next_r = R.from_quat(next[3:7])
            curr_r = R.from_quat(obs_robot_pos[3:7])
            next_euler = next_r.as_euler('zyz',degrees=False)
            curr_euler = curr_r.as_euler('zyz',degrees=False)
            action_rot  = next_euler-curr_euler


            action_grip = np.array([next[-1]  - obs_robot_pos[-1]])


            if action_rot[0]>2.0:
                action_rot[0] -=np.pi*2
            elif action_rot[0]<-2.0:
                action_rot[0] += np.pi * 2

            if action_rot[1]>2.0:
                action_rot[1] -=np.pi*2
            elif action_rot[1]<-2.0:
                action_rot[1] += np.pi * 2

            if action_rot[2]>2.0:
                action_rot[2] -=np.pi*2
            elif action_rot[2]<-2.0:
                action_rot[2] += np.pi * 2


            action = np.concatenate((action_pos,action_rot,action_grip))
            pre_obs = _flatten_obs(obs, obs_robot_pos_list)
            # if sum(abnormal_buffer) < 0.0002: #0.000000000000001:
            #     # action = np.random.rand(7)*0.15
            #     action[2] = 1
            #     action[5] = 0.0
            #     # print("haha")
            #     for m in range(5):
            #         obs, reward, done, _ = env.step(action)
            #         obs_robot = _flatten_obs(obs, obs_robot_list)
            #         abnormal_buffer.append(abs(sum(obs_robot[:3] - pre_obs[:3])))
            #         obs_obj = _flatten_obs(obs, obs_obj_list)
            #         obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)
            #         one_state = np.concatenate((obs_robot_pos, obs_obj))
            #         current_subgoal = np.array([get_current_stage(one_state)])
            #         env.render()
            #     action[2] = 0
            #     action[5] = 0.1
            #     print("haha")
            #     for m in range(5):
            #         obs, reward, done, _ = env.step(action)
            #         obs_robot = _flatten_obs(obs, obs_robot_list)
            #         abnormal_buffer.append(abs(sum(obs_robot[:3] - pre_obs[:3])))
            #         obs_obj = _flatten_obs(obs, obs_obj_list)
            #         obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)
            #         one_state = np.concatenate((obs_robot_pos, obs_obj))
            #         current_subgoal = np.array([get_current_stage(one_state)])
            #         env.render()
            #
            # else:
            #     obs, reward, done, _ = env.step(action)
            obs, reward, done, _ = env.step(action)





            obs_robot = _flatten_obs(obs, obs_robot_list)
            abnormal_buffer.append(abs(sum(obs_robot[:3]-pre_obs[:3])))

            obs_obj = _flatten_obs(obs, obs_obj_list)

            obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)
            one_state = np.concatenate((obs_robot_pos, obs_obj))
            current_subgoal = np.array([get_current_stage(one_state)])

            if current_subgoal == 3:
                count +=1
            if count >= 1:
                current_subgoal = np.array([3])
            env.render()
            # if abs(obs["cubeA_pos"][0]-obs["cubeB_pos"][0]) < 0.025 and abs(obs["cubeA_pos"][1]-obs["cubeB_pos"][1]) < 0.025 and abs(obs["cubeA_pos"][2]-obs["cubeB_pos"][2])<0.05:
            #     current_subgoal = np.array([3])
            #     print("==============done===============")
            #     print(obs["cubeA_pos"])
            #     print(obs["cubeB_pos"])

            print(current_subgoal)

        obs = env.reset()
        obs_robot = _flatten_obs(obs, obs_robot_list)
        obs_obj = _flatten_obs(obs, obs_obj_list)

        obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)

        one_state = np.concatenate((obs_robot_pos, obs_obj))
        current_subgoal = np.array([get_current_stage(one_state)])

