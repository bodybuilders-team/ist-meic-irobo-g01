#!/usr/bin/env python2.7

import rospy
import rosbag
import sys
import argparse
import numpy
from tf2_ros import Buffer, TransformListener
from rosgraph_msgs.msg import Clock
import matplotlib.pyplot as plt
import numpy as np
import math

from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovariance
from nav_msgs.msg import Odometry
import tf2_py as tf2
import tf2_geometry_msgs
import sympy as sp

# a, b, x0, y0, x1, y1, theta, alpha = sp.symbols("a b x0 y0 x1 y1 theta alpha")

# # Calculated by applying the rotation matrix to the parametric equations of an ellipse
# x_rot = (a*sp.cos(theta)*sp.cos(alpha)) - (b*sp.sin(theta)*sp.sin(alpha)) + x0 
# y_rot = (a*sp.cos(theta)*sp.sin(alpha)) + (b*sp.sin(theta)*sp.cos(alpha)) + y0

# d_squared = ((x_rot - x1)**2)+((y_rot-y1)**2)

# d_squared_derivative = d_squared.diff(theta)


tf_buffer: Buffer = None
fixed_frame = "mocap"
time_data = []
error_data = []
initial_time = None
gt_frame = None
gt_frame = None
est_pose_data = []
ellipse_data = []
gt_pose_data = []
lower_bound_pose_data = []
upper_bound_pose_data = []


# def extrema_to_ellipse(gt_tf: TransformStamped, uncertainty_ellipse: tuple):
#     (a, b, rotation, x0, y0) = uncertainty_ellipse
#     gt_trans = gt_tf.transform.translation
#     (x1, y1) = (gt_trans.x, gt_trans.y)
#     print(gt_trans)
#     d_squared_derivative_substituted = d_squared_derivative.subs({x0:x0,y0:y0,a:a,b:b,x1:x1,y1:y1, alpha:rotation})
#     solutions = sp.solve(d_squared_derivative_substituted, theta)

#     print(f"Solutions = {solutions}")
#     return ( )

def extrema_to_ellipse(gt_tf: TransformStamped, uncertainty_ellipse: tuple):
    (a, b, alpha, x0, y0) = uncertainty_ellipse
    gt_trans = gt_tf.transform.translation
    (x1, y1) = (gt_trans.x, gt_trans.y)
    # Generate points on the ellipse before rotation
    theta = np.linspace(0, 2 * np.pi, 1000)

    global x_rot, y_rot
    
    # Rotate the ellipse alpha degrees counterclockwise
    x_rot = (a*np.cos(theta)*np.cos(alpha)) - (b*np.sin(theta)*np.sin(alpha)) + x0 
    y_rot = (a*np.cos(theta)*np.sin(alpha)) + (b*np.sin(theta)*np.cos(alpha)) + y0

    # Calculate the distances between point a and the points on the ellipse
    distances = np.sqrt((x_rot - x1)**2 + (y_rot - y1)**2)
    min_i = np.argmin(distances)

    min_x = x_rot[min_i]
    min_y = y_rot[min_i]

    max_i = np.argmax(distances)

    max_x = x_rot[max_i]
    max_y = y_rot[max_i]

    min_pose = PoseStamped()
    min_pose.pose.position.x = min_x
    min_pose.pose.position.y = min_y

    max_pose = PoseStamped()
    max_pose.pose.position.x = max_x
    max_pose.pose.position.y = max_y

    return (min_pose, max_pose)

def calculate_error(gt_tf: TransformStamped, pose: PoseStamped):
    dx = gt_tf.transform.translation.x - pose.pose.position.x
    dy = gt_tf.transform.translation.y - pose.pose.position.y
    return np.sqrt(dx**2 + dy**2)

def calculate_bounded_error(gt_tf, pose, uncertainty_ellipse, bf2bs_tf):
    (lower_bound_pose, higher_bound_pose) = extrema_to_ellipse(gt_tf, uncertainty_ellipse)

    lower_bound_pose_data.append([lower_bound_pose.pose.position.x, lower_bound_pose.pose.position.y])
    upper_bound_pose_data.append([higher_bound_pose.pose.position.x, higher_bound_pose.pose.position.y])

    pose = tf2_geometry_msgs.do_transform_pose(pose, bf2bs_tf)
    lower_bound_pose = tf2_geometry_msgs.do_transform_pose(lower_bound_pose, bf2bs_tf)
    higher_bound_pose = tf2_geometry_msgs.do_transform_pose(higher_bound_pose, bf2bs_tf)

    error = calculate_error(gt_tf, pose) *1e3
    lower_bound_error = calculate_error(gt_tf, lower_bound_pose) * 1e3
    higher_bound_error = calculate_error(gt_tf, higher_bound_pose) * 1e3

    return (lower_bound_error, error, higher_bound_error)

def odometry_callback(data: Odometry):
    global tf_buffer, gt_frame, initial_time, time_data, error_data, est_pose_data, gt_pose_data, ellipse_data

    gt_tf: TransformStamped = tf_buffer.lookup_transform(gt_frame, fixed_frame,time=rospy.Time(0)) #TODO: must be sourced from the odom frame time stamp
    bf2bs_tf = tf_buffer.lookup_transform(data.child_frame_id, "base_scan", time=rospy.Time(0), timeout=rospy.Duration(0.01))

    matrix = np.mat(data.pose.covariance)
    matrix = matrix.reshape(6,6)
    matrix = matrix[:2, :2]
    eig = np.linalg.eig(matrix)
    eigenvalues, eigenvectors = eig
    eigenorder = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[eigenorder], eigenvectors[:, eigenorder]
    
    pose: PoseWithCovariance = data.pose

    a, b = tuple(2*np.sqrt(eigenvalues)) # 95% probability
    x0, y0 = (pose.pose.position.x, pose.pose.position.y)
    ellipse_rotation = np.arctan2(*eigenvectors[:, 0][::-1]).item()
    uncertainty_ellipse = (a, b, ellipse_rotation, x0, y0) # a, b, x0, y0

    bounded_error = calculate_bounded_error(gt_tf, pose, uncertainty_ellipse, bf2bs_tf)

    curr_time = rospy.get_time()

    if(bounded_error[0] > bounded_error[1] or bounded_error[2] < bounded_error[1]):
        print("ALERTTTTTT: ", bounded_error)
    else:
        time_data.append(curr_time-initial_time)
        error_data.append(bounded_error)
        est_pose_data.append([pose.pose.position.x, pose.pose.position.y])
        gt_pose_data.append([gt_tf.transform.translation.x, gt_tf.transform.translation.y])


def main():
    global tf_buffer, gt_frame, initial_time, time_data, error_data, est_pose_data, gt_pose_data, ellipse_data, x_rot, y_rot
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_frame', help='The child frame of the GT transform', default='mocap_laser_link')
    parser.add_argument('--est_topic', help='The odometry topic of the estimation', default='/odometry/filtered')

    args = parser.parse_args()

    gt_frame = args.gt_frame
    est_topic = args.est_topic

    rospy.init_node('evaluation_node')

    if rospy.rostime.is_wallclock():
        rospy.logfatal('You should be using simulated time: rosparam set use_sim_time true')
        sys.exit(1)

    rospy.loginfo('Waiting for clock')
    rospy.sleep(0.00001)

    rospy.loginfo('Listening to frames and computing error, press S to stop')

    initial_time = rospy.get_time()

    tf_buffer = Buffer(cache_time=rospy.Duration(20))
    tf_listener = TransformListener(tf_buffer)

    sub = rospy.Subscriber(est_topic, Odometry, odometry_callback)

    print(tf_buffer)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    # plt.plot([est_pose_data[10][0]], [est_pose_data[10][1]])
    # plt.plot([gt_pose_data[10][0]] , [gt_pose_data[10][1]])
    # plt.show() 

    plt.title("Elipse at time %.2f seconds" % time_data[-1])
    plt.ylabel("y coordinate (m)")
    plt.xlabel("x coordinate (m)")
    plt.axis('equal')
    plt.plot(x_rot, y_rot, label='Covariance ellipse')
    print("\n Timestamp: ", time_data[-1] + initial_time)
    print("\n Ground thruth: ", est_pose_data[-1][0], est_pose_data[-1][1])
    plt.plot(gt_pose_data[-1][0] , [gt_pose_data[-1][1]], 'ro', label='Ground truth')
    print("\n Estimated ", est_pose_data[-1][0] , est_pose_data[-1][1])
    plt.plot(est_pose_data[-1][0], [est_pose_data[-1][1]], 'go', label='Estimated position')
    print("\n Lower bound: ", error_data[-1][0])
    plt.plot(lower_bound_pose_data[-1][0], [lower_bound_pose_data[-1][1]], 'yx', label='Lower bound')
    print("\n Upper bound: ", error_data[-1][2])
    plt.plot(upper_bound_pose_data[-1][0], [upper_bound_pose_data[-1][1]], 'cx', label='Upper bound')
    plt.legend(loc='best')
    plt.show() 

    plt.title("Mocap vs Estimation Error")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Time (Seconds)")
    plt.plot(time_data, error_data)
    plt.show() 

if __name__ == "__main__":
    main()