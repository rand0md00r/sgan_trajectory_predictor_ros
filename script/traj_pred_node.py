# -*- coding: utf-8 -*-

#!/home/enter/envs/torch04/bin/python
import sys
sys.path.append('/home/work_space/src/traj_pred/')

import time
import rospy
import torch
import numpy as np
import collections

from attrdict import AttrDict

from models.models import TrajectoryGenerator
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import random
import random

# Global variables
DEBUG = False
MAX_CONSIDERED_PEDS = 10
corlor_list = []
for i in range(MAX_CONSIDERED_PEDS):
    corlor_list.append([round(random.uniform(0, 1), 1) for _ in range(3)])


class twoDimQueue:
    def __init__(self) -> None:
        self.queue_2d = collections.deque([], maxlen=MAX_CONSIDERED_PEDS) 
        for i in range(MAX_CONSIDERED_PEDS):        
            self.queue_2d.appendleft(collections.deque([[0.0, 0.0]]*8, maxlen=8)) # 10个“8个（0.0， 0.0）”

    def is_empty(self):
        return len(self.queue_2d) == 0
    
    def size(self):
        return len(self.queue_2d)
    
    def new_person(self):
        self.queue_2d.append(self.queue_zero_steps)
        
    def update(self, namespase=str, position_x=float, position_y=float):
        ns_int = int(namespase)
        ns_int = ns_int % MAX_CONSIDERED_PEDS
        self.queue_2d[ns_int].append([position_x, position_y])


class MarkerArrayCallbackClass:
    def __init__(self, pub_pred, pub_obs):
        self.sample_time = 0
        self.obs_traj = torch.Tensor()              # (8, max_ped_nums, 2) 观测轨迹
        self.obs_traj_rel = torch.Tensor()          # (8, max_ped_nums, 2) 观测轨迹的相对坐标   
        self.seq_start_end = []                     # [(0, max_ped_nums)]  观测轨迹的起始点和终止点
        self.pred_traj_fake_rel = torch.Tensor()    # (8, max_ped_nums, 2) 预测轨迹的相对坐标
        self.pred_traj_fake = torch.Tensor()
        self.generator = self.loadModel()
        
        self.frame_data = twoDimQueue()

        self.pub_pred = pub_pred
        self.pub_obs = pub_obs
        
        rospy.loginfo("Waiting for the MOT Box Topic...")

    def loadModel(self):
        # 加载预训练的PyTorch模型
        model_path = '/home/work_space/src/traj_pred/models/checkpoint/sgan-p-models/eth_8_model.pt'
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
    
    
    def publish_marker_array(self, publisher, data_3d):
        # data = self.pred_traj_fake.detach().cpu().numpy()
        data = data_3d.detach().cpu().numpy()
        marker_array = MarkerArray()

        max_pos_num = data.shape[0]     # 预测的步数 = 8
        max_ped_num = data.shape[1]     # 每帧预测的行人数

        # 每个行人一个marker
        for ped_idx in range(max_ped_num):
            
            marker = Marker()
            marker.header.frame_id = "os_sensor"
            marker.ns = "prediction" if publisher == self.pub_pred else "observation"
            marker.type = marker.SPHERE_LIST
            marker.action = marker.ADD
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.1
            
            marker.color.a = 1.0
            marker.color.r = corlor_list[ped_idx][0] if publisher == self.pub_pred else 1.0
            marker.color.g = corlor_list[ped_idx][1] if publisher == self.pub_pred else 1.0
            marker.color.b = corlor_list[ped_idx][2] if publisher == self.pub_pred else 1.0
            marker.pose.orientation.w = 1.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0

            for steps_idx in range(max_pos_num):
                point = Point()
                point.x = data[steps_idx, ped_idx, 0]
                point.y = data[steps_idx, ped_idx, 1]
                point.z = 0
                marker.id = max_pos_num * (ped_idx) + steps_idx  # 设置唯一的id
                marker.points.append(point)

            marker_array.markers.append(marker)
        publisher.publish(marker_array)


    def get_obs_traj(self, markarray):
        # 确定当前帧的人数 max_ped_nums
        peds_in_curr_seq = [int(marker.ns) % MAX_CONSIDERED_PEDS for marker in markarray.markers]

        if peds_in_curr_seq:
            max_ped_nums = max(peds_in_curr_seq) + 1
            if max_ped_nums > MAX_CONSIDERED_PEDS:
                rospy.logwarn("Observed pedestrians exceed the maximum number, only consider the first %s pedestrians.", MAX_CONSIDERED_PEDS)
        else:
            max_ped_nums = 0
            rospy.logwarn("no peds in current seq")
            return


        # 更新当前帧frame_data
        for marker in markarray.markers:
            self.frame_data.update(marker.ns, marker.pose.position.x, marker.pose.position.y)
        
        # 根据 frame_data 和 max_ped_nums 确定obs_traj
        obs_traj_np = np.zeros((8, max_ped_nums, 2))
        obs_traj_rel_np = np.zeros((8, max_ped_nums, 2))

        for cur_ped_name in range(max_ped_nums):
            obs_traj_np[:, cur_ped_name, :] = np.array(self.frame_data.queue_2d[cur_ped_name])

        obs_traj_np = np.around(obs_traj_np, decimals=4)  # (8, max_ped_nums + 1, 2)
        obs_traj_rel_np[1:, :, :] = obs_traj_np[1:, :, :] - obs_traj_np[:-1, :, :]

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


    def markarray_callback(self, markarray):
        # 1. 获取obs_traj
        # TODO obs_traj检查，每个位置的值是否对应
        self.get_obs_traj(markarray)

        # 2. 加载模型,得到模型输出
        if self.obs_traj is not None and len(self.obs_traj.shape) > 1:
            rospy.logdebug("Observed %s pedestrians: ", self.obs_traj.shape[1])
            self.pred_traj_fake_rel = self.generator(self.obs_traj, self.obs_traj_rel, self.seq_start_end)
        else:
            # 无obs_traj时，不进行预测
            rospy.logwarn("obs_traj is None or obs_traj.shape is wrong.")
            return

        # 3. 计算绝对坐标，发布模型输出
        if self.relative_to_abs(self.pred_traj_fake_rel, self.obs_traj[-1], self.pred_traj_fake) == False:
            rospy.logwarn("Relative to abs failed.")
        else:
            rospy.loginfo("Predicted %s pedestrian trajectories.", self.pred_traj_fake.shape[1])
        self.publish_marker_array(self.pub_pred, self.pred_traj_fake)
        self.publish_marker_array(self.pub_obs, self.obs_traj)



if __name__ == '__main__':
    if DEBUG == True:
        mot_box_topic = 'motion_markers'
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.DEBUG)
    else:
        mot_box_topic = '/mot_tracking/box'
        rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.INFO)
    
    
    pub_pred = rospy.Publisher('traj_pred_node/predictions', MarkerArray, queue_size=8)
    pub_obs = rospy.Publisher('traj_pred_node/observations', MarkerArray, queue_size=8)
    
    MarkerArrayCallbackClass = MarkerArrayCallbackClass(pub_pred, pub_obs)
    

    rospy.Subscriber(mot_box_topic, MarkerArray, MarkerArrayCallbackClass.markarray_callback)
    
    

    rospy.spin()
