import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from Model.Model import IGL, InvDyn_add, IGL_large
import torch

def get_current_stage(one_state):
    flag = 0

    if sum((one_state[:3] - one_state[9:12]) ** 2) < 0.01 and abs(one_state[7]<0.025):
        flag += 1
    if sum((one_state[:3] - one_state[9:12]) ** 2) < 0.01 and abs(one_state[7]<0.025) and abs(one_state[11] - one_state[-5])>0.045:
        flag += 1
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
    # current_subgoal = np.array([get_current_stage(one_state)])
    current_subgoal = np.array([0])

    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    all_dim = 24 # 9 + 13 + 1
    robot_dim = 9
    igl = IGL_large(all_dim, robot_dim, 'cpu')
    igl.load_state_dict(torch.load('./model_save/IGL_sg_imp210'))

    state_dim = 32
    next_state_dim = 9
    action_dim = 7

    Inv = InvDyn_add(state_dim,next_state_dim, action_dim, 'cpu')
    # Inv.load_state_dict(torch.load('./model_save/InvDyn_4.pth'))

    igl.eval()
    Inv.eval()
    from collections import deque

    abnormal = deque(maxlen=4)
    from scipy.spatial.transform import Rotation as R
    key = " "
    while True:
        obs_robot_traj = []
        obs_obj_traj   = []
        next_traj      = []
        for i in range(300):
            one_state = np.concatenate((one_state, current_subgoal))
            next_=igl(torch.FloatTensor(one_state).unsqueeze(0))
            next = next_.squeeze(0).detach().numpy()
            action_pos = np.array([(next[0]-obs_robot_pos[0]),(next[1]-obs_robot_pos[1]),(next[2]-obs_robot_pos[2])])*10

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
            action_rot *= 0
            action = np.concatenate((action_pos,action_rot,action_grip))


            obs, reward, done, _ = env.step(action)

            obs_robot_traj.append(obs_robot_pos.tolist())
            obs_obj_traj.append(obs_obj.tolist())
            next_traj.append(next.tolist())

            obs_robot = _flatten_obs(obs, obs_robot_list)
            obs_obj = _flatten_obs(obs, obs_obj_list)

            obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)
            one_state = np.concatenate((obs_robot_pos, obs_obj))
            current_subgoal = np.array([get_current_stage(one_state)])
            current_subgoal = np.array([0])


            env.render()
        obs_robot_traj = np.array(obs_robot_traj)
        obs_obj_traj   = np.array(obs_obj_traj)
        next_traj      = np.array(next_traj)

        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation as R

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scale = 500
        x, y, z = zip(*obs_robot_traj[:,:3])
        ax.scatter(x, y, z, color='b', alpha=1.0)
        r = R.from_quat(obs_robot_traj[:,3:7])
        # ==z==
        U, V, W = zip(*(r.as_matrix()[:, :, 2] / scale))
        ax.quiver(x, y, z, U, V, W, color='b')
        # ==y==
        U, V, W = zip(*(r.as_matrix()[:, :, 1] / scale))
        ax.quiver(x, y, z, U, V, W, color='g')
        # ==x==
        U, V, W = zip(*(r.as_matrix()[:, :, 0] / scale))
        ax.quiver(x, y, z, U, V, W, color='r')

        x, y, z = zip(*obs_obj_traj[:,:3])
        ax.scatter(x, y, z, color='m', alpha=1.0)
        r = R.from_quat(obs_obj_traj[:,3:7])
        # ==z==
        U, V, W = zip(*(r.as_matrix()[:, :, 2] / scale))
        ax.quiver(x, y, z, U, V, W, color='b')
        # ==y==
        U, V, W = zip(*(r.as_matrix()[:, :, 1] / scale))
        ax.quiver(x, y, z, U, V, W, color='g')
        # ==x==
        U, V, W = zip(*(r.as_matrix()[:, :, 0] / scale))
        ax.quiver(x, y, z, U, V, W, color='r')

        x, y, z = zip(*next_traj[:,:3])
        ax.scatter(x, y, z, color='c', alpha=1.0)
        r = R.from_quat(next_traj[:,3:7])
        # ==z==
        U, V, W = zip(*(r.as_matrix()[:, :, 2] / scale))
        ax.quiver(x, y, z, U, V, W, color='b')
        # ==y==
        U, V, W = zip(*(r.as_matrix()[:, :, 1] / scale))
        ax.quiver(x, y, z, U, V, W, color='g')
        # ==x==
        U, V, W = zip(*(r.as_matrix()[:, :, 0] / scale))
        ax.quiver(x, y, z, U, V, W, color='r')



        defal = 0.01
        ax.set_xlim([-defal + x[-1], defal + x[-1]])
        ax.set_ylim([-defal + y[-1], defal + y[-1]])
        ax.set_zlim([-defal + z[-1], defal + z[-1]])
        ax.set_xlabel('X___')
        ax.set_ylabel('Y___')
        ax.set_zlabel('Z___')
        plt.show()
        raise
        obs = env.reset()
        obs_robot = _flatten_obs(obs, obs_robot_list)
        obs_obj = _flatten_obs(obs, obs_obj_list)

        obs_robot_pos = _flatten_obs(obs, obs_robot_pos_list)

        one_state = np.concatenate((obs_robot_pos, obs_obj))
        current_subgoal = np.array([get_current_stage(one_state)])

