import pickle
import os
import numpy as np
import math
import quaternion
from Common.Interpolate_traj import traj_interpolation_stack

def sub_goal_separator(sub_goal):
    count = 0
    idx_list = []
    for idx in range(len(sub_goal)):
        if sub_goal[idx] != sub_goal[idx+1]:
            count += 1
            idx_list.append(idx)
            if count == 4:
                break
        else:
            pass
    return idx_list



data_concat = []
for pickle_data in os.listdir(os.getcwd()+'/data_IGL'):
    if 'data_IGL_sg4_' in pickle_data:
        with open('./data_IGL/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

traj_sg0 = []
traj_sg1 = []
traj_sg2 = []
traj_sg3 = []
traj_sg4 = []
print(len(data_concat))
coefs = np.linspace(0,1,3,endpoint=True)
for i in range(len(data_concat)-1):
    for j in range(i+1,len(data_concat)):
        choice=np.array([i,j])

        obs_robot1 = np.array(data_concat[choice[0]]['obs_robot'])
        obs_robot2 = np.array(data_concat[choice[1]]['obs_robot'])
        obs_obj1 = np.array(data_concat[choice[0]]['obs_obj'])
        obs_obj2 = np.array(data_concat[choice[1]]['obs_obj'])

        sub_goal1 = np.array(data_concat[choice[0]]['sg'])
        sub_goal2 = np.array(data_concat[choice[1]]['sg'])
        idx_sub_traj1 = sub_goal_separator(sub_goal1)
        idx_sub_traj2 = sub_goal_separator(sub_goal2)


        for k in range(len(idx_sub_traj1)+1):
            if k == 0:
                robot_candi1 = obs_robot1[:idx_sub_traj1[k]+1]
                robot_candi2 = obs_robot2[:idx_sub_traj2[k]+1]
                obj_candi1   = obs_obj1[:idx_sub_traj1[k]+1]
                obj_candi2   = obs_obj2[:idx_sub_traj2[k]+1]
            elif k == (len(idx_sub_traj1)):
                robot_candi1 = obs_robot1[idx_sub_traj1[k-1]+1:]
                robot_candi2 = obs_robot2[idx_sub_traj2[k-1]+1:]
                obj_candi1   = obs_obj1[idx_sub_traj1[k-1]+1:]
                obj_candi2   = obs_obj2[idx_sub_traj2[k-1]+1:]
            else:
                robot_candi1 = obs_robot1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                robot_candi2 = obs_robot2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]
                obj_candi1   = obs_obj1[idx_sub_traj1[k-1]+1:idx_sub_traj1[k]+1]
                obj_candi2   = obs_obj2[idx_sub_traj2[k-1]+1:idx_sub_traj2[k]+1]

            if i == 0 and j == 1:
                for coef in coefs:
                    fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,k,coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                    # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                    if k == 0:
                        traj_sg0.append(fixed_traj)
                    elif k == 1:
                        traj_sg1.append(fixed_traj)
                    elif k == 2:
                        traj_sg2.append(fixed_traj)
                    elif k == 3:
                        traj_sg3.append(fixed_traj)
                    elif k == 4:
                        traj_sg4.append(fixed_traj)
            elif i == 0 and j != 1:
                for coef in coefs[1:]:
                    fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,k,coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                    # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                    if k == 0:
                        traj_sg0.append(fixed_traj)
                    elif k == 1:
                        traj_sg1.append(fixed_traj)
                    elif k == 2:
                        traj_sg2.append(fixed_traj)
                    elif k == 3:
                        traj_sg3.append(fixed_traj)
                    elif k == 4:
                        traj_sg4.append(fixed_traj)
            else:
                for coef in coefs[1:-1]:
                    fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,k,coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
                    # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
                    if k == 0:
                        traj_sg0.append(fixed_traj)
                    elif k == 1:
                        traj_sg1.append(fixed_traj)
                    elif k == 2:
                        traj_sg2.append(fixed_traj)
                    elif k == 3:
                        traj_sg3.append(fixed_traj)
                    elif k == 4:
                        traj_sg4.append(fixed_traj)


print(len(traj_sg0))
print(len(traj_sg1))
print(len(traj_sg2))
print(len(traj_sg3))
print(len(traj_sg4))
with open('./data_IGL/Inter_mid_sg0.pickle', 'wb') as f:
    pickle.dump(traj_sg0, f, pickle.HIGHEST_PROTOCOL)
with open('./data_IGL/Inter_mid_sg1.pickle', 'wb') as f:
    pickle.dump(traj_sg1, f, pickle.HIGHEST_PROTOCOL)
with open('./data_IGL/Inter_mid_sg2.pickle', 'wb') as f:
    pickle.dump(traj_sg2, f, pickle.HIGHEST_PROTOCOL)
with open('./data_IGL/Inter_mid_sg3.pickle', 'wb') as f:
    pickle.dump(traj_sg3, f, pickle.HIGHEST_PROTOCOL)
with open('./data_IGL/Inter_mid_sg4.pickle', 'wb') as f:
    pickle.dump(traj_sg4, f, pickle.HIGHEST_PROTOCOL)
