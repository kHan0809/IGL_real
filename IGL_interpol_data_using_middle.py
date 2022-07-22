import pickle
import os
import numpy as np
import math
import quaternion
from Common.Interpolate_traj import traj_interpolation_stack




data_concat = []
subgoal = '3'
for pickle_data in os.listdir(os.getcwd()+'/data_IGL'):
    # if 'IGL' in pickle_data:
    if 'Inter_mid_sg'+subgoal in pickle_data:
        with open('./data_IGL/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

All_traj = []
coefs = np.linspace(0,1,4,endpoint=True)
print(coefs)
print(len(data_concat))
# middle_point = len(data_concat)//2
middle_point = 1
for i in range(len(data_concat)-1):
    for j in range(i+middle_point,len(data_concat)):
        choice=np.array([i,j])

        obs_robot1 = np.array(data_concat[choice[0]]['obs_robot'])
        obs_robot2 = np.array(data_concat[choice[1]]['obs_robot'])
        obs_obj1 = np.array(data_concat[choice[0]]['obs_obj'])
        obs_obj2 = np.array(data_concat[choice[1]]['obs_obj'])

        sub_goal1 = np.array(data_concat[choice[0]]['sg'])
        sub_goal2 = np.array(data_concat[choice[1]]['sg'])



        robot_candi1 = obs_robot1
        robot_candi2 = obs_robot2
        obj_candi1   = obs_obj1
        obj_candi2   = obs_obj2
        if i == 0 and j == middle_point:
            for coef in coefs:
                fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)
        elif i == 0 and j != middle_point:
            for coef in coefs[1:]:
                fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)
        else:
            for coef in coefs[1:-1]:
                fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                All_traj.append(fixed_traj)


print(len(All_traj))
with open('./data_IGL/Inter_using_mid_sg'+subgoal+'.pickle', 'wb') as f:
    pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)