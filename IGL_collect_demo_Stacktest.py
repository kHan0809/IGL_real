"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import pickle
from scipy.spatial.transform import Rotation as R
def _flatten_obs(obs_dict, name_list):
    ob_lst = []
    for key in name_list:
        if key in obs_dict:
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst)

def my_input2action():
    key = input()
    if 'a' in key:
        action = np.array([0., -0.8, 0., 0., 0., 0., -1.0])
    if 'd' in key:
        action = np.array([0., 0.8, 0., 0., 0., 0., -1.0])
    if 'w' in key:
        action = np.array([-0.8, 0.,  0., 0., 0., 0., -1.0])
    if 's' in key:
        action = np.array([0.8, 0.,  0., 0., 0., 0., -1.0])
    if 'r' in key:
        action = np.array([0., 0.,  0.8, 0., 0., 0., -1.0])
    if 'f' in key:
        action = np.array([0., 0., -0.8, 0., 0., 0., -1.0])

    if 'z' in key:
        action = np.array([0., 0.,  0., -0.2, 0., 0., -1.0])
    if 'x' in key:
        action = np.array([0., 0.,  0., 0.2, 0., 0., -1.0])
    if 'c' in key:
        action = np.array([0., 0.,  0., 0., 0., -0.2, -1.0])
    if 'v' in key:
        action = np.array([0., 0.,  0., 0., 0., 0.2, -1.0])
    if 't' in key:
        action = np.array([0., 0.,  0., 0.,  0.2, 0., -1.0])
    if 'g' in key:
        action = np.array([0., 0.,  0., 0., -0.2, 0., -1.0])
    if 'm' in key:
        action = np.array([0., 0., 0., 0., 0.0, 0., 1.0])
    if ',' in key:
        action = np.array([0., 0., 0., 0., 0.0, 0., -1.0])
    if 'n' in key:
        action *= np.array([1.0, 1.0,  1.0, 1.0, 1.0, 1.0, -1.0])

    # if 'a' in key:
    #     action = np.array([0.,-3.75, 0., 0., 0., 0., -1.0])
    # if 'd' in key:
    #     action = np.array([0., 3.75, 0., 0., 0., 0., -1.0])
    # if 'w' in key:
    #     action = np.array([-3.75, 0.,  0., 0., 0., 0., -1.0])
    # if 's' in key:
    #     action = np.array([3.75, 0.,  0., 0., 0., 0., -1.0])
    # if 'r' in key:
    #     action = np.array([0., 0.,  3.75, 0., 0., 0., -1.0])
    # if 'f' in key:
    #     action = np.array([0., 0., -3.75, 0., 0., 0., -1.0])
    #
    # if 'z' in key:
    #     action = np.array([0., 0.,  0., -0.15, 0., 0., -1.0])
    # if 'x' in key:
    #     action = np.array([0., 0.,  0., 0.15, 0., 0., -1.0])
    # if 'c' in key:
    #     action = np.array([0., 0.,  0., 0., 0., -0.15, -1.0])
    # if 'v' in key:
    #     action = np.array([0., 0.,  0., 0., 0., 0.15, -1.0])
    # if 't' in key:
    #     action = np.array([0., 0.,  0., 0.,  0.15, 0., -1.0])
    # if 'g' in key:
    #     action = np.array([0., 0.,  0., 0., -0.15, 0., -1.0])
    # if 'm' in key:
    #     action = np.array([0., 0., 0., 0., 0.0, 0., 1.0])
    # if ',' in key:
    #     action = np.array([0., 0., 0., 0., 0.0, 0., -1.0])
    # if 'n' in key:
    #     action *= np.array([1.0, 1.0,  1.0, 1.0, 1.0, 1.0, -1.0])

    return action
from scipy.spatial.transform import Rotation as R
def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    obs = env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    obs_robot_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    obs_obj_list  = ["cubeA_pos", "cubeA_quat","cubeB_pos","cubeB_quat"]

    ep_obs_robot, ep_obs_obj, ep_subgoal ,ep_action = [], [], [], []

    obs_robot = _flatten_obs(obs, obs_robot_list)
    obs_obj   = _flatten_obs(obs, obs_obj_list)
    one_state = np.concatenate((obs_robot, obs_obj))
    current_subgoal = get_current_stage(one_state,0,obs)

    ep_obs_robot.append(obs_robot)
    ep_obs_obj.append(obs_obj)
    ep_subgoal.append(current_subgoal)
    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        # action, grasp = input2action(
        #     device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        # )
        try:
            action = my_input2action()
        except:
            action = my_input2action()

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment stepa

        quat = [obs['robot0_eef_quat'][1],obs['robot0_eef_quat'][2],obs['robot0_eef_quat'][3],obs['robot0_eef_quat'][0]]
        r_b = R.from_quat(quat)
        print("================")
        print('act',action)
        print('orig quat',obs['robot0_eef_quat'])
        print('--- quat', quat)
        print('Euler zyz',r_b.as_euler('zyz',degrees=False))
        print('Euler zyx', r_b.as_euler('zyx', degrees=False))
        # print('pos',obs["robot0_eef_pos"])
        obs, reward, done, info = env.step(action)
        quat = [obs['robot0_eef_quat'][1], obs['robot0_eef_quat'][2], obs['robot0_eef_quat'][3],obs['robot0_eef_quat'][0]]
        r = R.from_quat(quat)
        print('orig quat', obs['robot0_eef_quat'])
        print('--- quat', quat)
        print('Euler zyz',r.as_euler('zyz',degrees=False))
        print('Euler zyx', r.as_euler('zyx', degrees=False))
        # print('pos', obs["robot0_eef_pos"])
        print('zyz')
        print(r.as_euler('zyz',degrees=False) - r_b.as_euler('zyz',degrees=False))
        print(r_b.as_euler('zyz', degrees=False) - r.as_euler('zyz', degrees=False))
        print('zyx')
        print(r.as_euler('zyx',degrees=False) - r_b.as_euler('zyx',degrees=False))
        print(r_b.as_euler('zyx', degrees=False) - r.as_euler('zyx', degrees=False))


        obs_robot = _flatten_obs(obs, obs_robot_list)
        obs_obj = _flatten_obs(obs, obs_obj_list)
        one_state = np.concatenate((obs_robot, obs_obj))
        current_subgoal = get_current_stage(one_state,current_subgoal,obs)

        ep_obs_robot.append(obs_robot)
        ep_obs_obj.append(obs_obj)
        ep_subgoal.append(current_subgoal)
        print(current_subgoal)

        ep_action.append(action)

        env.render()


        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()
    episode = dict(obs_robot=ep_obs_robot,obs_obj=ep_obs_obj,sg=ep_subgoal,a=ep_action)
    return episode

def get_current_stage(one_state,curt_subgoal,obs):
    flag = curt_subgoal
    inner = 0.0075
    inner2 = 0.02
    inner_pro = 1.0
    if abs(one_state[0] - one_state[9]) < inner and abs(one_state[1] - one_state[10]) < inner and abs(one_state[2] - one_state[11]) < inner and flag == 0:
        flag += 1
    if abs(one_state[0] - one_state[9]) < inner2*inner_pro and abs(one_state[1] - one_state[10]) < inner2*inner_pro and abs(one_state[2] - one_state[11]) < inner2*inner_pro and abs(one_state[7]<0.0215) and flag == 1:
        flag += 1
    if abs(one_state[0] - one_state[9]) < inner2*inner_pro and abs(one_state[1] - one_state[10]) < inner2*inner_pro and abs(one_state[2] - one_state[11]) < inner2*inner_pro and abs(one_state[7]<0.0215) and abs(one_state[11] - one_state[-5])>0.070 and flag==2:
        flag += 1
    if abs(obs["cubeA_pos"][0]-obs["cubeB_pos"][0]) < 0.025 and abs(obs["cubeA_pos"][1]-obs["cubeB_pos"][1]) < 0.025 and abs(obs["cubeA_pos"][2]-obs["cubeB_pos"][2])<0.07 and flag==3:
        flag += 1
    return flag

obs_list = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cubeA_pos", "cubeA_quat","cubeB_pos"]

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Stack_with_site")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    total_epi = []
    while True:
        epi=collect_human_trajectory(env, device, args.arm, args.config)
        total_epi.append(epi)

        with open('data_IGL_sg4_4.pickle', 'wb') as f:
            pickle.dump(total_epi, f, pickle.HIGHEST_PROTOCOL)
        # gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
