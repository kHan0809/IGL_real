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
            if count == 2:
                break
        else:
            pass
    return idx_list




data_concat = []
for pickle_data in os.listdir(os.getcwd()+'/data_IGL'):
    if 'data_IGL_sg4' in pickle_data:
    # if 'middle' in pickle_data:
        with open('./data_IGL/'+ pickle_data, 'rb') as f:
            data = pickle.load(f)
            data_concat.extend(data)
    else:
        pass

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

All_traj = []
coefs = np.linspace(0,1,11,endpoint=True)
print(len(data_concat))
for i in range(1,len(data_concat)-1):
    for j in range(i+1,len(data_concat)):
        choice=np.array([i,j])

        obs_robot1 = np.array(data_concat[choice[0]]['obs_robot'])
        obs_robot2 = np.array(data_concat[choice[1]]['obs_robot'])
        obs_obj1 = np.array(data_concat[choice[0]]['obs_obj'])
        obs_obj2 = np.array(data_concat[choice[1]]['obs_obj'])

        sub_goal1 = np.array(data_concat[choice[0]]['sg'])
        sub_goal2 = np.array(data_concat[choice[1]]['sg'])
        # idx_sub_traj1 = sub_goal_separator(sub_goal1)
        # idx_sub_traj2 = sub_goal_separator(sub_goal2)
        # print(".............")
        # obs_robot1 = np.array(data_concat[71]['obs_robot'])[:5]
        # print(obs_robot1)
        # obs_robot1 = np.array(data_concat[75]['obs_robot'])[:5]
        # print(obs_robot1)


        robot_candi1 = obs_robot1
        robot_candi2 = obs_robot2
        obj_candi1   = obs_obj1
        obj_candi2   = obs_obj2
        for coef in coefs:
            print("=====================")
            fixed_traj = traj_interpolation_stack(robot_candi1,robot_candi2,obj_candi1,obj_candi2,sub_goal1[0],coef) # 여기 sub goal은 계속 바뀌어야 된다~~0 나중에 바꿔주셈
            # print("==========================")
            # # print(np.array(fixed_traj["obs_robot"])[-1,:3])
            # # print(np.array(fixed_traj["obs_obj"])[-1,:3])
            # # print(robot_candi1[-2:-1, :3] - obj_candi1[-2:-1, :3])
            # # print(robot_candi2[-2:-1, :3] - obj_candi2[-2:-1, :3])
            #
            #
            # # fixed_traj = fix_traj(new_traj,robot_candi1,robot_candi2,coef)
            # print(np.array(fixed_traj["obs_robot"])[-1, :3])
            # print(np.array(fixed_traj["obs_obj"])[-1, :3])
            # print(np.array(fixed_traj["obs_robot"])[-1, :3]-np.array(fixed_traj["obs_obj"])[-1, :3])
            #
            #
            # print(np.array(fixed_traj["obs_obj"])[-1, :3]-np.array(fixed_traj["obs_robot"])[-1,:3])

            num = -2
            # temp = np.array(fixed_traj["obs_robot"])
            # print(temp[num:,3].reshape(-1,1).shape)
            # print(temp[num:, 4:7].shape)
            # print(np.hstack((temp[num:, 4:7],temp[num:, 3].reshape(-1,1))).shape)
            #
            # new = np.hstack((temp[num:, 4:7], temp[num:, 3].reshape(-1, 1)))
            #
            # x,y,z = zip(*np.array(fixed_traj["obs_robot"])[num:,:3])
            # ax.scatter(x, y, z, color='m', alpha=1.0)
            # r = R.from_quat(new)
            # # ==z==
            # U, V, W = zip(*(r.as_matrix()[:, :, 2] / 200))
            # ax.quiver(x, y, z, U, V, W, color='b')
            # # ==y==
            # U, V, W = zip(*(r.as_matrix()[:, :, 1] / 200))
            # ax.quiver(x, y, z, U, V, W, color='g')
            # # ==x==
            # U, V, W = zip(*(r.as_matrix()[:, :, 0] / 200))
            # ax.quiver(x, y, z, U, V, W, color='r')

            x,y,z = zip(*np.array(fixed_traj["obs_robot"])[num:,:3])
            ax.scatter(x, y, z, color='m', alpha=1.0)
            r = R.from_quat(np.array(fixed_traj["obs_robot"])[num:,3:7])
            # ==z==
            U, V, W = zip(*(r.as_matrix()[:, :, 2] / 200))
            ax.quiver(x, y, z, U, V, W, color='b')
            # ==y==
            U, V, W = zip(*(r.as_matrix()[:, :, 1] / 200))
            ax.quiver(x, y, z, U, V, W, color='g')
            # ==x==
            U, V, W = zip(*(r.as_matrix()[:, :, 0] / 200))
            ax.quiver(x, y, z, U, V, W, color='r')


            x,y,z = zip(*np.array(fixed_traj["obs_obj"])[num:,:3])
            ax.scatter(x,y,z,color='r',alpha=0.5)
            print(np.array(fixed_traj["obs_obj"])[-1:,3:7])
            r = R.from_quat(np.array(fixed_traj["obs_obj"])[num:,3:7])
            #==z==
            U,V,W=zip(*(r.as_matrix()[:,:,2]/180))
            ax.quiver(x,y,z,U,V,W,color='b')
            #==y==
            U,V,W=zip(*(r.as_matrix()[:,:,1]/180))
            ax.quiver(x,y,z,U,V,W,color='g')
            #==x==
            U,V,W=zip(*(r.as_matrix()[:,:,0]/180))
            ax.quiver(x,y,z,U,V,W,color='r')



            x, y, z = zip(*np.array(obj_candi1[num:, :3]))
            ax.scatter(x, y, z, color='g', alpha=1.0)
            r = R.from_quat(obj_candi1[num:, 3:7])
            #==z==
            U,V,W=zip(*(r.as_matrix()[:,:,2]/180))
            ax.quiver(x,y,z,U,V,W,color='b')
            #==y==
            U,V,W=zip(*(r.as_matrix()[:,:,1]/180))
            ax.quiver(x,y,z,U,V,W,color='g')
            #==x==
            U,V,W=zip(*(r.as_matrix()[:,:,0]/180))
            ax.quiver(x,y,z,U,V,W,color='r')

            x, y, z = zip(*np.array(obj_candi2[num:, :3]))
            ax.scatter(x, y, z, color='b', alpha=1.0)
            print(obj_candi2[-1:, 3:7])
            r = R.from_quat(obj_candi2[num:, 3:7])
            #==z==
            U,V,W=zip(*(r.as_matrix()[:,:,2]/180))
            ax.quiver(x,y,z,U,V,W,color='b')
            #==y==
            U,V,W=zip(*(r.as_matrix()[:,:,1]/180))
            ax.quiver(x,y,z,U,V,W,color='g')
            #==x==
            U,V,W=zip(*(r.as_matrix()[:,:,0]/180))
            ax.quiver(x,y,z,U,V,W,color='r')


            # print("==========")
            # print(np.array(fixed_traj["obs_obj"])[num:,:])
            # print(obj_candi2[num:,:])

            defal = 0.08

            ax.set_xlim([-defal+x[0], defal+x[0]])
            ax.set_ylim([-defal+y[0], defal+y[0]])
            ax.set_zlim([-defal+z[0], defal+z[0]])
            ax.set_xlabel('X___')
            ax.set_ylabel('Y___')
            ax.set_zlabel('Z___')
            All_traj.append(fixed_traj)

        x, y, z = zip(*np.array(robot_candi1[num:, :3]))
        ax.scatter(x, y, z, color='b', alpha=1.0)
        r = R.from_quat(robot_candi1[num:, 3:7])
        # ==z==
        U, V, W = zip(*(r.as_matrix()[:, :, 2] / 200))
        ax.quiver(x, y, z, U, V, W, color='b')
        # ==y==
        U, V, W = zip(*(r.as_matrix()[:, :, 1] / 200))
        ax.quiver(x, y, z, U, V, W, color='g')
        # ==x==
        U, V, W = zip(*(r.as_matrix()[:, :, 0] / 200))
        ax.quiver(x, y, z, U, V, W, color='r')


        x, y, z = zip(*np.array(robot_candi2[num:, :3]))
        ax.scatter(x, y, z, color='b', alpha=1.0)
        r = R.from_quat(robot_candi2[num:, 3:7])
        # ==z==
        U, V, W = zip(*(r.as_matrix()[:, :, 2] / 200))
        ax.quiver(x, y, z, U, V, W, color='b')
        # ==y==
        U, V, W = zip(*(r.as_matrix()[:, :, 1] / 200))
        ax.quiver(x, y, z, U, V, W, color='g')
        # ==x==
        U, V, W = zip(*(r.as_matrix()[:, :, 0] / 200))
        ax.quiver(x, y, z, U, V, W, color='r')


        plt.show()
        raise
        x, y, z = zip(*robot_candi1[:,:3])
        ax.scatter(x, y, z, color='b')
        x, y, z = zip(*robot_candi2[:,:3])
        ax.scatter(x, y, z, color='c')
        plt.show()
        raise


print(len(All_traj))
# with open('./data_IGL/Inter_traj_using_mid.pickle', 'wb') as f:
#     pickle.dump(All_traj, f, pickle.HIGHEST_PROTOCOL)

#=========================그래프 그리는거야~~===================#
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# z_axis_scale = 80
# X, Y, Z = zip(*robot_candi1[:,:3])
# ax.scatter(X,Y,Z)
# r = R.from_quat(robot_candi1[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W)
# # ax.quiver(X[-1],Y[-1],Z[-1],U[-1],V[-1],W[-1],color='r')
#
# X, Y, Z = zip(*robot_candi2[:,:3])
# ax.scatter(X,Y,Z,color='r')
# r = R.from_quat(robot_candi2[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W,color='r')
#
# X, Y, Z = zip(*new_traj[:,:3])
# ax.scatter(X,Y,Z,color='g')
# r = R.from_quat(new_traj[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W,color='g')
#
# X, Y, Z = zip(*fixed_traj[:,:3])
# ax.scatter(X,Y,Z,color='m')
# r = R.from_quat(fixed_traj[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W,color='m')
#
# plt.show()
#==================================================#