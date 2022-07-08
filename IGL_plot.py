import pickle
import numpy as np
import os

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

for pickle_data in os.listdir(os.getcwd()):
    if 'path' in pickle_data:
        with open(pickle_data, 'rb') as f:
            data = pickle.load(f)
    else:
        pass




obs_robot1 = np.array(data[0]['obs_robot'])
obs_robot2 = np.array(data[1]['obs_robot'])

obs_obj1 = np.array(data[0]['obs_obj'])
obs_obj2 = np.array(data[1]['obs_obj'])

sub_goal_1 = np.array(data[0]['sg'])
sub_goal_2 = np.array(data[1]['sg'])

idx_sub_traj1 = sub_goal_separator(sub_goal_1)
idx_sub_traj2 = sub_goal_separator(sub_goal_2)

robot_candi1 = obs_robot1[:idx_sub_traj1[0]+1]
robot_candi2 = obs_robot2[:idx_sub_traj2[0]+1]

obj_candi1 = obs_obj1[:idx_sub_traj1[0]+1]
obj_candi2 = obs_obj2[:idx_sub_traj2[0]+1]

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


print(robot_candi1[-2:-1,:3]-obj_candi1[-2:-1,:3])
print(robot_candi2[-2:-1,:3]-obj_candi2[-2:-1,:3])
X, Y, Z = zip(*robot_candi1[-2:-1,:3])
ax.scatter(X,Y,Z,color='r')
X, Y, Z = zip(*obj_candi1[-2:-1,:3])
ax.scatter(X,Y,Z,color='m')
# r = R.from_quat(robot_candi1[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W)
# ax.quiver(X[-1],Y[-1],Z[-1],U[-1],V[-1],W[-1],color='r')

X, Y, Z = zip(*robot_candi2[-2:-1,:3])
ax.scatter(X,Y,Z,color='g')
X, Y, Z = zip(*obj_candi2[-2:-1,:3])
ax.scatter(X,Y,Z,color='c')
# r = R.from_quat(robot_candi2[:, 3:7])
# U,V,W=zip(*(r.as_matrix()[:,:,2]/z_axis_scale))
# ax.quiver(X,Y,Z,U,V,W,color='r')
plt.show()


