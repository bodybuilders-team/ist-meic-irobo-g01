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


tf_buffer: Buffer = None
time_data = []
error_data = []
initial_time = None
est_pose_data = []
ellipse_data = []
gt_pose_data = []
lower_bound_pose_data = []
upper_bound_pose_data = []
x_rot_data = []
y_rot_data = []

def extrema_to_ellipse(gt_tf: TransformStamped, uncertainty_ellipse: tuple):
    (a, b, alpha, x0, y0) = uncertainty_ellipse
    gt_trans = gt_tf.transform.translation
    (x1, y1) = (gt_trans.x, gt_trans.y)
    # Generate points on the ellipse before rotation
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Rotate the ellipse alpha degrees counter tf_listener = clockwise
    x_rot = (a*np.cos(theta)*np.cos(alpha)) - (b*np.sin(theta)*np.sin(alpha)) + x0 
    y_rot = (a*np.cos(theta)*np.sin(alpha)) + (b*np.sin(theta)*np.cos(alpha)) + y0

    x_rot_data.append(x_rot)
    y_rot_data.append(y_rot)

    # Calculate the distances between point a and the points on the ellipse
    distances = np.sqrt((x_rot - x1)**2 + (y_rot - y1)**2)
    min_i = np.argmin(distances)

    min_x = x_rot[min_i]
    min_y = y_rot[min_i]

    max_i = np.argmax(distances)

    max_x = x_rot[max_i]
    max_y = y_rot[max_i]

    min_pose = TransformStamped()
    min_pose.transform.translation.x = min_x
    min_pose.transform.translation.y = min_y

    max_pose = TransformStamped()
    max_pose.transform.translation.x = max_x
    max_pose.transform.translation.y = max_y

    return (min_pose, max_pose, x_rot, y_rot)

def calculate_distance(gt_tf: TransformStamped, est_tf: TransformStamped):
    dx = gt_tf.transform.translation.x - est_tf.transform.translation.x
    dy = gt_tf.transform.translation.y - est_tf.transform.translation.y
    return np.sqrt(dx**2 + dy**2)

def calculate_bounded_error(gt_tf, est_tf, uncertainty_ellipse):
    global tf_buffer, gt_frame, initial_time, time_data, error_data, est_pose_data, gt_pose_data, ellipse_data
    (lower_bound_pose, higher_bound_pose, x_rot, y_rot) = extrema_to_ellipse(gt_tf, uncertainty_ellipse)

    error = calculate_distance(gt_tf, est_tf) *1e3
    lower_bound_error = calculate_distance(gt_tf, lower_bound_pose) * 1e3
    higher_bound_error = calculate_distance(gt_tf, higher_bound_pose) * 1e3
    
    return ((lower_bound_pose, higher_bound_pose), (x_rot, y_rot), (lower_bound_error, error, higher_bound_error))

def is_within_ellipse(tf: TransformStamped, ellipse) -> bool:
    x, y = tf.transform.translation.x, tf.transform.translation.y
    a, b, theta, x0, y0 = ellipse
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_hat = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_hat = -sin_theta * (x - x0) + cos_theta * (y - y0)
    return (x_hat / a) ** 2 + (y_hat / b) ** 2 <= 1

def odometry_callback(data: Odometry):
    global tf_buffer, initial_time, time_data, error_data, est_pose_data, gt_pose_data, ellipse_data, x_rot_data, y_rot_data

    try:
        gt_tf: TransformStamped = tf_buffer.lookup_transform("mocap", "mocap_laser_link", time=data.header.stamp) #TODO: must be sourced from the odom frame time stamp
        # Get time from data
        est_tf: TransformStamped = tf_buffer.lookup_transform("mocap", "base_scan",time= data.header.stamp)
    except tf2.ExtrapolationException:
        return
    
    curr_time = rospy.get_time()

    matrix = np.mat(data.pose.covariance)
    matrix = matrix.reshape(6,6)
    matrix = matrix[:2, :2]
    eig = np.linalg.eig(matrix)
    eigenvalues, eigenvectors = eig
    eigenorder = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[eigenorder], eigenvectors[:, eigenorder]
    
    a, b = tuple(2*np.sqrt(eigenvalues)) # 95% probability
    x0, y0 = (est_tf.transform.translation.x, est_tf.transform.translation.y)
    ellipse_rotation = np.arctan2(*eigenvectors[:, 0][::-1]).item()
    uncertainty_ellipse = (a, b, ellipse_rotation, x0, y0) # a, b, x0, y0

    (bound_pose, ellipse_rot, bounded_error) = calculate_bounded_error(gt_tf, est_tf, uncertainty_ellipse)

    (x_rot, y_rot) =ellipse_rot
    (lower_bound_pose, higher_bound_pose) = bound_pose

    (lower_bound_error, error, higher_bound_error) = bounded_error
    
    # TODO: use within ellipse function and not within circle...
    if(calculate_distance(gt_tf, est_tf) < a): # Assuming ellipse is a circle...
        lower_bound_pose = gt_tf
        lower_bound_error = 0
        print("WITHIN ELLIPSE")
    
    if(lower_bound_error > error or higher_bound_error < error):
        print("ALERTTTTTT")

    time_data.append(curr_time-initial_time)
    est_pose_data.append([est_tf.transform.translation.x, est_tf.transform.translation.y])
    gt_pose_data.append([gt_tf.transform.translation.x, gt_tf.transform.translation.y])
    error_data.append((lower_bound_error, error, higher_bound_error))
    lower_bound_pose_data.append([lower_bound_pose.transform.translation.x, lower_bound_pose.transform.translation.y])
    upper_bound_pose_data.append([higher_bound_pose.transform.translation.x, higher_bound_pose.transform.translation.y])
    x_rot_data.append(x_rot)
    y_rot_data.append(y_rot)


def main():
    global tf_buffer, gt_frame, initial_time, time_data, error_data, est_pose_data, gt_pose_data, ellipse_data, x_rot_data, y_rot_data

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

    sub = rospy.Subscriber("odometry/filtered", Odometry, odometry_callback, buff_size=1)

    print(tf_buffer)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    # plt.plot([est_pose_data[10][0]], [est_pose_data[10][1]])
    # plt.plot([gt_pose_data[10][0]] , [gt_pose_data[10][1]])
    # plt.show() 

    # TODO: Change to uncertainty graph
    plt.title("Elipse at time %.2f seconds" % time_data[-1])
    plt.ylabel("y coordinate (m)")
    plt.xlabel("x coordinate (m)")
    plt.axis('equal')
    plt.plot(x_rot_data[-1], y_rot_data[-1], label='Covariance ellipse')
    print("\n Timestamp: ", time_data[-1] + initial_time)
    print("\n Ground thruth: ", gt_pose_data[-1][0], gt_pose_data[-1][1])
    plt.plot(gt_pose_data[-1][0] , [gt_pose_data[-1][1]], 'ro', label='Ground truth')
    print("\n Estimated ", est_pose_data[-1][0] , est_pose_data[-1][1])
    plt.plot(est_pose_data[-1][0], [est_pose_data[-1][1]], 'go', label='Estimated position')
    print("\n Lower bound: ", error_data[-1][0])
    plt.plot(lower_bound_pose_data[-1][0], [lower_bound_pose_data[-1][1]], 'yx', label='Lower bound')
    print("\n Upper bound: ", error_data[-1][2])
    plt.plot(upper_bound_pose_data[-1][0], [upper_bound_pose_data[-1][1]], 'cx', label='Upper bound')
    plt.legend(loc='best')
    plt.show() 

    lower_bound_errors = [e[0] for e in error_data]
    errors = [e[1] for e in error_data]
    upper_bound_errors = [e[2] for e in error_data]

    plt.title("Mocap vs Estimation Error")
    plt.ylabel("Distance (mm)")
    plt.xlabel("Time (Seconds)")
    
    # TODO: Change
    # Plot each error type with a specific label
    plt.plot(time_data, lower_bound_errors, label='Lower bound error')
    plt.plot(time_data, errors, label='Error')
    plt.plot(time_data, upper_bound_errors, label='Upper bound error')
    
    plt.legend(loc='best')
    plt.show() 

if __name__ == "__main__":
    main()