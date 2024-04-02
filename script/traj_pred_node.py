#! /home/sgan/env/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/work_space/src/traj_pred/')

import argparse
import time
import rospy
import torch
import numpy as np
import collections
import copy
from sklearn.cluster import DBSCAN

from attrdict import AttrDict

from models.models import TrajectoryGenerator
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import random

# Global variables
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='/home/work_space/src/traj_pred/models/checkpoint/sgan-p-models/eth_8_model.pt', help='path for loading trained models.')
parser.add_argument('--max_peds',   type=int, default=15, help='max considered peds.')
parser.add_argument('--debug',      type=str, default='False', help='debug mode.')
parser.add_argument('--pred_topic', type=str, default='traj_pred_node/predictions', help='topic of predictions.')
parser.add_argument('--obs_topic',  type=str, default='traj_pred_node/observations', help='topic of observations.')
parser.add_argument('--esti_topic', type=str, default='traj_pred_node/estimations',  help='topic of estimations.')
parser.add_argument('--frame',      type=str, default='camera_init', help='frame of the markers.')

parserArgs = parser.parse_args()

corlor_list = []
for i in range(parserArgs.max_peds):
    corlor_list.append([round(random.uniform(0, 1), 1) for _ in range(3)])


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, motion_model, observation_model, motion_noise, observation_noise):
        self.state = initial_state
        self.covariance_P = initial_covariance
        self.motion_model_A = motion_model
        self.observation_model_H = observation_model
        self.motion_noise_Q = motion_noise
        self.observation_noise_R = observation_noise
        self.dt = 0.4

    def predict(self):
        # Motion prediction
        predicted_state = self.motion_model_A @ self.state
        predicted_covariance_P = self.motion_model_A @ self.covariance_P @ self.motion_model_A.T + self.motion_noise_Q
        self.state = predicted_state
        self.covariance_P = predicted_covariance_P

    def correct(self, measurement):
        # Measurement update (correction)
        kalman_gain = self.covariance_P @ self.observation_model_H.T @ np.linalg.inv(self.observation_model_H @ self.covariance_P @ self.observation_model_H.T + self.observation_noise_R)
        measurement_diff = measurement.reshape(-1, 1) - self.observation_model_H @ self.state
        self.state += kalman_gain @ measurement_diff
        self.covariance_P = (np.eye(len(self.state)) - kalman_gain @ self.observation_model_H) @ self.covariance_P
        

class frameData:
    def __init__(self) -> None:
        # self.queue_2d = collections.deque([], maxlen=parserArgs.max_peds) 
        # for i in range(parserArgs.max_peds):        
            # self.queue_2d.appendleft(collections.deque([[0.0, 0.0]]*8, maxlen=8)) # 10个“8个（0.0， 0.0）”
        self.traj_dict = {}

    def is_empty(self):
        return len(self.traj_dict) == 0
    
    def size(self):
        return len(self.traj_dict)
    
    def add_new_person(self, new_peds_ids=list, steps = 8):
        for ped_id in new_peds_ids:
            self.traj_dict[ped_id] = collections.deque([[0.0, 0.0, 0.0]]*steps, maxlen=steps)
        
    def del_lost_person(self, lost_peds_ids=list):
        for ped_id in lost_peds_ids:
            self.traj_dict.pop(ped_id)
        
    def update(self, id=str, position_x=float, position_y=float):
        # ns_int = int(float(namespase))
        # ns_int = ns_int % parserArgs.max_peds
        # self.queue_2d[ns_int].append([position_x, position_y])
        self.traj_dict[id].append([position_x, position_y, 0.0])
        
    def get_pose_len(self):
        for it in self.traj_dict.keys():
            return len(self.traj_dict[it])
        return 0

    def calculate_angle_between_points(self, point1, point2):
        """
        计算两个二维点之间的角度
        """
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        angle = np.arctan2(delta_y, delta_x)
        return np.degrees(angle)
        
    def fillTheta(self):
        if len(self.traj_dict) != 0:
            for peds_idx in self.traj_dict.keys():
                traj = self.traj_dict[peds_idx]
                for i in range(len(traj) - 1):
                    point1 = np.array(traj[i])
                    point2 = np.array(traj[i+1])
                    angle = self.calculate_angle_between_points(point1, point2)
                    # 将角度添加到轨迹中
                    self.traj_dict[peds_idx][i][-1] = np.deg2rad(angle)
                if len(traj) > 1:
                    self.traj_dict[peds_idx][-1][-1] = self.traj_dict[peds_idx][-2][-1]
                    


class MarkerArrayCallbackClass:
    def __init__(self, pub_pred, pub_esti, pub_obs):
        self.sample_time = 0
        self.obs_traj = torch.Tensor()              # (8, max_ped_nums, 2) 观测轨迹
        self.obs_traj_rel = torch.Tensor()          # (8, max_ped_nums, 2) 观测轨迹的相对坐标   
        self.seq_start_end = []                     # [(0, max_ped_nums)]  观测轨迹的起始点和终止点
        self.pred_traj_fake_rel = torch.Tensor()    # (8, max_ped_nums, 2) 预测轨迹的相对坐标
        self.pred_traj_fake = torch.Tensor()
        self.generator = self.loadModel()
        
        self.markarray = None
        self.curr_peds = []
        self.last_obs_peds = []
        self.lost_peds = []
        self.new_peds = []

        self.frame_data = frameData()
        
        # post process
        self.post_process = True
        self.pred_frame_data = frameData()
        self.estimation_frame_data = frameData()
        self.dt = 0.4
        self.motion_model = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.observation_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.initial_state = np.array([[0], [0], [0], [0]])  # Initial state (x, y, theta)
        self.initial_covariance = np.eye(4) * 0.1  # Initial covariance
        self.motion_noise = np.eye(4) * 0.001
        self.observation_noise = np.eye(2) * 0.1
        
        # self.kf = KalmanFilter(self.initial_state, 
        #                        self.initial_covariance, 
        #                        self.motion_model, 
        #                        self.observation_model, 
        #                        self.motion_noise, 
        #                        self.observation_noise)
        
        self.pub_pred = pub_pred
        self.pub_esti = pub_esti
        self.pub_obs = pub_obs

        self.timer = rospy.Timer(rospy.Duration(0.4), self.timer_callback)
        
        rospy.loginfo("Waiting for the MOT Topic...")

    def loadModel(self):
        # 加载预训练的PyTorch模型
        rospy.loginfo("Loading model...")
        model_path = parserArgs.model_name
        checkpoint = torch.load(model_path)
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
                obs_len=args.obs_len,
                pred_len=args.pred_len,
                embedding_dim=args.embedding_dim,
                encoder_h_dim=args.encoder_h_dim_g,
                decoder_h_dim=args.decoder_h_dim_g,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                noise_dim=args.noise_dim,
                noise_type=args.noise_type,
                noise_mix_type=args.noise_mix_type,
                pooling_type=args.pooling_type,
                pool_every_timestep=args.pool_every_timestep,
                dropout=args.dropout,
                bottleneck_dim=args.bottleneck_dim,
                neighborhood_size=args.neighborhood_size,
                grid_size=args.grid_size,
                batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        # if torch.cuda.is_available():
        generator.cuda()
        generator.train()
        rospy.loginfo("Model loaded.")
        return generator


    def relative_to_abs(self, rel_traj, start_pos, pred_traj_fake):
        # 将相对坐标转换为绝对坐标的代码
        if rel_traj is None or start_pos is None:
            return False
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
        self.pred_traj_fake = abs_traj.permute(1, 0, 2)
        
        return True

    def DBSCAN_fillter(trajectory, eps=0.1, min_samples=2):
        # 转换轨迹为NumPy数组
        data = np.array(trajectory)
        
        # 使用DBSCAN算法识别离群点
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # 创建一个字典来存储每个簇的索引
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # 填充离群点
        filled_trajectory = []
        for label, cluster_indices in clusters.items():
            if label == -1:  # 如果是离群点
                for index in cluster_indices:
                    if index == 0:
                        filled_value = data[1]
                    elif index == len(data) - 1:
                        filled_value = data[-2]
                    else:
                        filled_value = (data[index-1] + data[index+1]) / 2  # 使用周围点的均值来填充
                    filled_trajectory.append(filled_value)
            else:  # 如果不是离群点
                for index in cluster_indices:
                    filled_trajectory.append(data[index])
        
        return filled_trajectory

    
    def publish_marker_array(self, publisher, flag=0):
        
        if flag == 0:
            frame_data = self.pred_frame_data
            ns = "predition"
        elif flag == 1:
            frame_data = self.estimation_frame_data
            ns = "estimation"
        elif flag == 2:
            frame_data = self.frame_data
            ns = "observation"
        marker_array = MarkerArray()

        max_ped_num = frame_data.size()         # 每帧预测的行人数
        max_pos_num = frame_data.get_pose_len()   # 预测的步数 = 8
        
        i = 0
        # 每个行人一个marker
        for ped_idx in frame_data.traj_dict.keys():
            i = i +1
            marker = Marker()
            marker.header.frame_id = "os_sensor"
            marker.ns = ns
            marker.type = marker.CUBE_LIST
            marker.action = marker.ADD
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 1.0
            
            marker.color.a = 0.5
            # marker.color.r = corlor_list[i][0]
            # marker.color.g = corlor_list[i][1]
            # marker.color.b = corlor_list[i][2]
            marker.color.r = 1.0 if ns == "predition" else 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0 if ns == "predition" else 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0

            for steps_idx in range(max_pos_num):
                point = Point()
                point.x = frame_data.traj_dict[ped_idx][steps_idx][0]
                point.y = frame_data.traj_dict[ped_idx][steps_idx][1]
                point.z = 0
                marker.id = max_pos_num * (ped_idx) + steps_idx  # 设置唯一的id
                marker.points.append(point)

            marker_array.markers.append(marker)
        publisher.publish(marker_array)
        
    def zeroSpeedCulculate(self, obs_traj):
        # 计算零速度
        # 1. 计算obs_traj的速度
        obs_traj_rel = obs_traj[1:, :, :] - obs_traj[:-1, :, :]
        obs_speed = np.linalg.norm(obs_traj_rel, axis=2)
        
        return np.around(obs_speed, 2)

    def findClosestPed(self, cur_pose):
        cur_ped_idx = None
        min_distance = 1000
        for ped_idx in self.frame_data.traj_dict.keys():
            distance = np.linalg.norm(np.array(self.frame_data.traj_dict[ped_idx][-1][:2]) - np.array(cur_pose))
            if distance < min_distance:
                cur_ped_idx = ped_idx
                min_distance = distance
        return cur_ped_idx
        
    
    def fillPredictedTraj(self):
        for ped_idx in self.frame_data.traj_dict.keys():
            self.pred_frame_data.add_new_person([ped_idx], steps=16)
            for i in range(len(self.frame_data.traj_dict[ped_idx])):
                self.pred_frame_data.traj_dict[ped_idx][i] = self.frame_data.traj_dict[ped_idx][i]
        
        for p_idx in range(self.pred_traj_fake.shape[1]):
            pose = [self.pred_traj_fake[-1, p_idx, 0].cpu().item(), self.pred_traj_fake[-1, p_idx, 1].cpu().item()]
            cur_ped_idx = self.findClosestPed(pose)

            for s_idx in range(8, 16):
                self.pred_frame_data.traj_dict[cur_ped_idx][s_idx] = [self.pred_traj_fake[s_idx - 8, p_idx, 0].cpu().item(), self.pred_traj_fake[s_idx - 8, p_idx, 1].cpu().item(), 0.0]

            if self.pred_frame_data.traj_dict[cur_ped_idx][0] != self.frame_data.traj_dict[cur_ped_idx][0]:
                print("cur_ped_idx:", cur_ped_idx)
                print("self.frame_data.traj_dict[cur_ped_idx]", self.frame_data.traj_dict[cur_ped_idx][0])
                print("self.pred_frame_data.traj_dict[cur_ped_idx]", self.pred_frame_data.traj_dict[cur_ped_idx][0]) 

        self.pred_frame_data.fillTheta()


    def get_obs_traj(self, markarray):
        if markarray == None:
            return
        # 确定当前帧行人id
        self.curr_peds = [int(float(marker.ns)) for marker in markarray.markers]
        rospy.logdebug("curr ped: %s", self.curr_peds)

        # 根据上一帧行人和当前帧行人，确定新出现的行人和消失的行人
        self.new_peds = list(set(self.curr_peds) - set(self.last_obs_peds))
        self.lost_peds = list(set(self.last_obs_peds) - set(self.curr_peds))
        self.last_obs_peds = self.curr_peds
        rospy.logdebug("new ped: %s", self.new_peds)
        rospy.logdebug("lost ped: %s", self.lost_peds)


        # 确定当前帧的人数 max_ped_nums
        if self.curr_peds:
            max_ped_nums = len(self.curr_peds)
            if max_ped_nums > parserArgs.max_peds:
                rospy.logwarn("Observed pedestrians exceed the maximum number, only consider the first %s peds, but obs %s peds", parserArgs.max_peds, max_ped_nums)
        else:
            max_ped_nums = 0
            rospy.logwarn("no peds in current seq")
            return

        # 根据lost和new，更新frame_data, 删、增、更新
        for ped_id in self.lost_peds:
            self.frame_data.del_lost_person([ped_id])
            
        for ped_id in self.new_peds:
            self.frame_data.add_new_person([ped_id])
            
        for marker in markarray.markers:
            self.frame_data.update(int(float(marker.ns)), marker.pose.position.x, marker.pose.position.y)

        # 根据 frame_data 和 max_ped_nums 确定obs_traj
        obs_traj_np = np.zeros((8, max_ped_nums, 2))
        obs_traj_rel_np = np.zeros((8, max_ped_nums, 2))

        for cur_ped_idx in range(max_ped_nums):
            real_ped_id = self.curr_peds[cur_ped_idx]

            obs_traj_np[:, cur_ped_idx, :] = np.array(self.frame_data.traj_dict[real_ped_id])[:, :2]

        obs_traj_np = np.around(obs_traj_np, 2)  # (8, max_ped_nums + 1, 2)
        obs_traj_rel_np[1:, :, :] = np.around(obs_traj_np[1:, :, :] - obs_traj_np[:-1, :, :], 2)

        self.seq_start_end = [(0, max_ped_nums)]


        if np.any(np.array(obs_traj_np.shape) < 0):
            rospy.logerr("obs_traj_np has negative dimensions.")
            return
        else:
            self.obs_traj = torch.from_numpy(obs_traj_np).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(obs_traj_rel_np).type(torch.float)
        self.seq_start_end = torch.tensor(self.seq_start_end)
        self.obs_traj = self.obs_traj.to('cuda')
        self.obs_traj_rel = self.obs_traj_rel.to('cuda')
        
        self.frame_data.fillTheta()


    def markarray_callback(self, markarray):
        self.markarray = markarray

    def timer_callback(self, event):
        # 1. 获取obs_traj
        # TODO obs_traj检查，每个位置的值是否对应
        self.get_obs_traj(self.markarray)

        # 2. 加载模型,得到模型输出
        if self.obs_traj is not None and len(self.obs_traj.shape) > 1:
            rospy.logdebug("Observed %s pedestrians: ", self.obs_traj.shape[1])
            self.pred_traj_fake_rel = self.generator(self.obs_traj, self.obs_traj_rel, self.seq_start_end)
        else:
            # 无obs_traj时，不进行预测
            rospy.logwarn("obs_traj is None or obs_traj.shape is wrong.")
            return
        
        # 3. 计算绝对坐标
        if self.relative_to_abs(self.pred_traj_fake_rel, self.obs_traj[-1], self.pred_traj_fake) == False:
            rospy.logwarn("Relative to abs failed.")
        else:
            rospy.loginfo("Predicted %s pedestrian trajectories.", self.pred_traj_fake.shape[1])
            
        # 4. 填充pred_frame_data
        self.fillPredictedTraj()
        self.estimation_frame_data = copy.deepcopy(self.pred_frame_data)       # 初始化估计轨迹

        if self.post_process == True:
            # 5. 使用Kalman滤波，估计预测轨迹
            for ped_idx in self.estimation_frame_data.traj_dict.keys():
                measurements = np.array(self.estimation_frame_data.traj_dict[ped_idx])
                measurements = measurements[:, :2].T
                estimated_states = np.zeros((2, self.estimation_frame_data.get_pose_len()))
                
                history_kf = KalmanFilter(self.initial_state, self.initial_covariance, self.motion_model, self.observation_model, self.motion_noise, self.observation_noise)                
                init_v_x = (self.estimation_frame_data.traj_dict[ped_idx][1][0] - self.estimation_frame_data.traj_dict[ped_idx][0][0]) / self.dt
                init_v_y = (self.estimation_frame_data.traj_dict[ped_idx][1][1] - self.estimation_frame_data.traj_dict[ped_idx][0][1]) / self.dt
                history_kf.state = np.array([[self.estimation_frame_data.traj_dict[ped_idx][0][0]], 
                                             [self.estimation_frame_data.traj_dict[ped_idx][0][1]], 
                                             [init_v_x],
                                             [init_v_y]])
                for t in range(8):
                    history_kf.predict()
                    history_kf.correct(measurements[:, t])
                    
                pred_kf = copy.deepcopy(history_kf)
                t_v_x = (self.estimation_frame_data.traj_dict[ped_idx][7][0] - self.estimation_frame_data.traj_dict[ped_idx][6][0]) / self.dt
                t_v_y = (self.estimation_frame_data.traj_dict[ped_idx][7][1] - self.estimation_frame_data.traj_dict[ped_idx][6][1]) / self.dt
                pred_kf.state = np.array([[self.estimation_frame_data.traj_dict[ped_idx][7][0]], 
                                          [self.estimation_frame_data.traj_dict[ped_idx][7][1]], 
                                          [t_v_x],
                                          [t_v_y]])
                for t in range(8, 16):
                    pred_kf.predict()
                    pred_kf.correct(measurements[:, t])
                    estimated_states[0, t] = pred_kf.state.squeeze()[0]
                    estimated_states[1, t] = pred_kf.state.squeeze()[1]
                    self.estimation_frame_data.traj_dict[ped_idx][t] = estimated_states[:, t]
            

            
            
        # 5. 发布marker
        # self.publish_marker_array(self.pub_pred, self.pred_traj_fake)
        # self.publish_marker_array(self.pub_obs, self.obs_traj)
        # 修改发布函数，使用frame_data中的数据
        self.publish_marker_array(self.pub_pred, flag=0)
        self.publish_marker_array(self.pub_esti, flag=1)


if __name__ == '__main__':
    if parserArgs.debug == 'True':
        # mot_box_topic = 'motion_markers'
        mot_box_topic = 'obstacles'
        parserArgs.frame = 'odom'
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.DEBUG)
    else:
        mot_box_topic = '/mot_tracking/box'
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.INFO)
    
    rospy.loginfo("model_name: %s", parserArgs.model_name)
    rospy.loginfo("max_peds: %s", parserArgs.max_peds)
    rospy.loginfo("debug: %s", parserArgs.debug)
    rospy.loginfo("pred_topic: %s", parserArgs.pred_topic)
    rospy.loginfo("obs_topic: %s", parserArgs.obs_topic)
    rospy.loginfo("mot_box_topic: %s", mot_box_topic)
    
    pub_pred = rospy.Publisher(parserArgs.pred_topic, MarkerArray, queue_size=8)
    pub_obs = rospy.Publisher(parserArgs.obs_topic, MarkerArray, queue_size=8)
    pub_esti = rospy.Publisher(parserArgs.esti_topic, MarkerArray, queue_size=8)
    
    MarkerArrayCallbackClass = MarkerArrayCallbackClass(pub_pred, pub_esti, pub_obs)
    

    rospy.Subscriber(mot_box_topic, MarkerArray, MarkerArrayCallbackClass.markarray_callback)
    
    

    rospy.spin()
