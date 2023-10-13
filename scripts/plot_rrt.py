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

from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovariance, Point
from nav_msgs.msg import Odometry
import tf2_py as tf2
import tf2_geometry_msgs
import sympy as sp


tf_buffer: Buffer = None

points = []

def point_callback(data: Point):
    points.append((data.x, data.y))
    if (len(points) == 2):
        # add line from points[0] to points[1]
        plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'b')


def main():
    rospy.init_node('evaluation_node')

    if rospy.rostime.is_wallclock():
        rospy.logfatal('You should be using simulated time: rosparam set use_sim_time true')
        sys.exit(1)

    rospy.loginfo('Waiting for clock')
    rospy.sleep(0.00001)

    rospy.loginfo('Listening to frames and computing error, press S to stop')

    sub = rospy.Subscriber("/move_base/RRTPlannerROS/global_plan_points", Point, point_callback, buff_size=1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    # TODO: Change to uncertainty graph
    plt.title("RRT")
    plt.ylabel("y coordinate (m)")
    plt.xlabel("x coordinate (m)")
    plt.axis('equal')
    plt.show() 

if __name__ == "__main__":
    main()