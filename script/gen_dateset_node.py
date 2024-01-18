# -*- coding: utf-8 -*-

#!/home/enter/envs/torch04/bin/python
import sys
sys.path.append('/home/work_space/src/traj_pred/')

import rospy
import numpy as np
import collections
import argparse

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import random

# Global variables
DEBUG = False
mot_box_topic = "/mot_tracking/box"
file_name = ""
output_path = "/home/work_space/src/traj_pred/dataset/raw/"
save_path = output_path + file_name


def process_arguments():
    parser = argparse.ArgumentParser(description='Process file name from command line.')
    parser.add_argument('file_name', type=str, help='Specify the file name.')

    args = parser.parse_args()
    return args.file_name

class MarkerArrayCallbackClass:
    def __init__(self):
        self.frame_idx = 0
        self.ped_idx = 0
        self.position_x = 0.0
        self.position_y = 0.0
        self.save_file = open(save_path, "w")

        rospy.loginfo("Waiting for the MOT Box Topic...")

    def save_current_frame(self, frame_idx=int, ped_idx=int, position_x=float, position_y=float):
        self.save_file.write(str(round(float(self.frame_idx), 1)) + " " + \
            str(round(float(self.ped_idx), 1)) + " " + \
            str(round(float(self.position_x), 2)) + " " + \
            str(round(float(self.position_y), 2)) + "\n")
        print(str(self.frame_idx) + " " + str(self.ped_idx) + " " + str(self.position_x) + " " + str(self.position_y))

    def markarray_callback(self, markarray):
        for marker in markarray.markers:
            self.ped_idx = marker.ns
            
            self.position_x = round(marker.pose.position.x, 2)
            self.position_y = round(marker.pose.position.y, 2)
            self.save_current_frame()    
        self.frame_idx += 1
        

    def __del__(self):
        self.save_file.close()
        print("The dataset has been saved to " + save_path)




if __name__ == '__main__':
    
    file_name = process_arguments()
    print('File name received:', file_name)
    save_path = output_path + file_name
    
    if DEBUG == True:
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.DEBUG)
    else:
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.INFO)
    
    
    MarkerArrayCallbackClass = MarkerArrayCallbackClass()
    

    rospy.Subscriber(mot_box_topic, MarkerArray, MarkerArrayCallbackClass.markarray_callback)
    
    

    rospy.spin()

