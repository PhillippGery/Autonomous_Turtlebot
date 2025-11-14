#!/usr/bin/env python3

import math
import sys
import os
import numpy as np

import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_share_directory

from PIL import Image
import yaml
import pandas as pd

from copy import copy
import time
from queue import PriorityQueue


# Import other python packages that you think necessary


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
    

def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()