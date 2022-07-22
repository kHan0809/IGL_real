import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from Model.Model import IGL, InvDyn_add, IGL_large, IGL_large_sep
import torch
from robosuite.wrappers import VisualizationWrapper
def get_current_stage(one_state,obs):
    flag = 0
    inner = 0.020
    inner_pro = 1.1
    if abs(one_state[0] - one_state[9]) < inner and abs(one_state[1] - one_state[10]) < inner and abs(one_state[2] - one_state[11]) < inner*0.75:
        flag = 1
    if abs(one_state[0] - one_state[9]) < inner*inner_pro and abs(one_state[1] - one_state[10]) < inner*inner_pro and abs(one_state[2] - one_state[11]) < inner*inner_pro and abs(one_state[7]<0.024):
        flag = 2
    if abs(one_state[0] - one_state[9]) < inner*inner_pro and abs(one_state[1] - one_state[10]) < inner*inner_pro and abs(one_state[2] - one_state[11]) < inner*inner_pro and abs(one_state[7]<0.024) and abs(one_state[11] - one_state[-5])>0.025:
        flag = 3
    if abs(obs["cubeA_pos"][0]-obs["cubeB_pos"][0]) < inner*1.0 and abs(obs["cubeA_pos"][1]-obs["cubeB_pos"][1]) < inner*1.0 and abs(obs["cubeA_pos"][2]-obs["cubeB_pos"][2])<0.07 and flag==3:
        flag = 4
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
    options["env_name"] = "Stack_with_site"

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
    current_subgoal = np.array([get_current_stage(one_state,obs)])
    # current_subgoal = np.array([0])

    env.viewer.set_camera(camera_id=0)
    env = VisualizationWrapper(env)

    # Get action limits
    low, high = env.action_spec

    all_dim = 24 # 9 + 13 + 1
    robot_dim = 9
    igl0 = IGL_large_sep(all_dim, robot_dim, 'cpu')
    igl0.load_state_dict(torch.load('./model_save/BEST/SEPv2_IGL_sg0_imp100'))
    igl1 = IGL_large_sep(all_dim, robot_dim, 'cpu')
    igl1.load_state_dict(torch.load('./model_save/BEST/SEP_IGL_sg1_imp001'))
    igl2 = IGL_large_sep(all_dim, robot_dim, 'cpu')
    igl2.load_state_dict(torch.load('./model_save/BEST/SEP_IGL_sg2_imp010'))
    igl3 = IGL_large_sep(all_dim, robot_dim, 'cpu')
    igl3.load_state_dict(torch.load('./model_save/SEP_IGL_sg3_imp_fine001'))
    igl4 = IGL_large_sep(all_dim, robot_dim, 'cpu')
    igl4.load_state_dict(torch.load('./model_save/BEST/SEP_IGL_sg4_imp100'))

    state_dim = 32
    next_state_dim = 9
    action_dim = 7

    # Inv = InvDyn_add(state_dim,next_state_dim, action_dim, 'cpu')
    # Inv.load_state_dict(torch.load('./model_save/InvDyn_4.pth'))


    igl0.eval()
    igl1.eval()
    igl2.eval()
    igl3.eval()
    igl4.eval()
    from scipy.spatial.transform import Rotation as R
    while True:
        End = False
        task_completion_hold_count = -1
        for i in range(1200):
            one_state = np.concatenate((one_state, current_subgoal))
            if current_subgoal == 0:
                next_= igl0(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 1:
                next_ = igl1(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 2:
                next_ = igl2(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 3:
                next_ = igl3(torch.FloatTensor(one_state).unsqueeze(0))
            elif current_subgoal == 4:
                next_ = igl4(torch.FloatTensor(one_state).unsqueeze(0))

            # action=Inv.forward(torch.FloatTensor(obs_robot).unsqueeze(0),next_)
            next = next_.squeeze(0).detach().numpy()
            action_pos = np.array([(next[0]-obs_robot_pos[0]),(next[1]-obs_robot_pos[1]),(next[2]-obs_robot_pos[2])])*3
            # if current_subgoal == 3:
            #     action_pos *= 2.0


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
            # print(action_rot)
            # if abs(action_rot[0]) >0.03:
            #     action_rot[0] /= 10
            # if abs(action_rot[2]) >0.3:
            #     action_rot[2] /= 10

            action = np.concatenate((action_pos,action_rot,action_grip))

            pre_obs = _flatten_obs(obs, obs_robot_pos_list)
            obs, reward, done, info = env.step(action)






            obs_robot = _flatten_obs(obs, obs_robot_list)

            obs_obj = _flatten_obs(obs, obs_obj_list)

            obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)
            one_state = np.concatenate((obs_robot_pos, obs_obj))
            current_subgoal = np.array([get_current_stage(one_state,obs)])
            print(current_subgoal)

            env.render()
            # input()
            if current_subgoal == 4:
                End = True
            if End:
                current_subgoal = np.array([4])


            # Also break if we complete the task
            if task_completion_hold_count == 0:
                break

            # state machine to check for having a success for 10 consecutive timesteps
            print(env._check_success())
            if env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = 10  # reset count on first success timestep
            else:
                task_completion_hold_count = -1  # null the counter if there's no success



        obs = env.reset()
        obs_robot = _flatten_obs(obs, obs_robot_list)
        obs_obj = _flatten_obs(obs, obs_obj_list)

        obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)

        one_state = np.concatenate((obs_robot_pos, obs_obj))
        current_subgoal = np.array([get_current_stage(one_state,obs)])
