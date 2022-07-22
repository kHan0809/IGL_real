import math
import numpy as np
def q2e(x,y,z,w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 -2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])

# def q2e(w,x,y,z):
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x * x + y * y)
#     roll_x = math.atan2(t0, t1)
#
#     t2 = 2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch_y = math.asin(t2)
#
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 -2.0 * (y * y + z * z)
#     yaw_z = math.atan2(t3, t4)
#
#     return np.array([roll_x, pitch_y, yaw_z])
