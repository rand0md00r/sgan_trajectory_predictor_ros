#!/usr/bin/env python
import sys
sys.path.append('/home/work_space/src/traj_pred/')

import time
import rospy
import torch
import numpy as np
import collections

from attrdict import AttrDict

from models.models import TrajectoryGenerator
from visualization_msgs.msg import MarkerArray
from trejectory import Trajectory

# 定义全局变量
trejector = Trajectory()
zero_queue = collections.deque([[0.0, 0.0]]*8, maxlen=8)
frame_data  = []
for i in range(10):
    frame_data.append(zero_queue)
obs_traj, obs_traj_rel, seq_start_end = torch.Tensor(), torch.Tensor(), []
pred_traj_fake_rel = torch.Tensor()

sample_time = 0


def test_relative_to_abs():
    # 创建输入数据
    rel_traj = torch.tensor([[[1, 1], [1, 1]], 
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 1]]
                             ])
    print(rel_traj.shape)
    print(rel_traj)
    start_pos = torch.tensor([1, 1])
    
    # 调用函数
    abs_traj = relative_to_abs(rel_traj, start_pos)

    # 打印结果
    print(abs_traj)
    
    exit()

def relative_to_abs(rel_traj, start_pos):
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

def get_obs_traj(markarray):    
    # 确定当前帧的人数 curr_id_nums
    peds_in_curr_seq = [marker.id for marker in markarray.markers]
    if peds_in_curr_seq:
        curr_id_nums = max(peds_in_curr_seq)
    else:
        curr_id_nums = 0
        print("No markers in the array.")

    # 队列不足时补充队列
    while(len(frame_data) < curr_id_nums):        
        for i in range(curr_id_nums):
            frame_data.append(zero_queue)
    
    # 更新当前帧frame_data
    for marker in markarray.markers:
        if marker.id <= len(frame_data):
            frame_data[marker.id].appendleft([marker.pose.position.x, marker.pose.position.x])
    
    # 根据 frame_data 和 curr_id_nums 确定obs_traj
    obs_traj_np = np.zeros((8, curr_id_nums, 2))
    obs_traj_rel_np = np.zeros((8, curr_id_nums, 2))
    
    for cur_ped in range(curr_id_nums):
        obs_traj_np[:, cur_ped, :] = np.array(frame_data[cur_ped])
        # obs_traj_rel_np[:, cur_ped, :] = np.array(frame_data[cur_ped]) - np.array(frame_data[cur_ped])[-1]
    
    obs_traj_np = np.around(obs_traj_np, decimals=4)  # (8, curr_id_nums, 2)
    obs_traj_rel_np[1:, :, :] = obs_traj_np[1:, :, :] - obs_traj_np[:-1, :, :]

    
    seq_start_end = [(0, curr_id_nums)]
    # obs_traj = torch.from_numpy(obs_traj_np).type(torch.float)
    if np.any(np.array(obs_traj_np.shape) < 0):
        rospy.logerr("obs_traj_np has negative dimensions.")
    else:
        obs_traj = torch.from_numpy(obs_traj_np).type(torch.float)
    obs_traj_rel = torch.from_numpy(obs_traj_rel_np).type(torch.float)
    seq_start_end = torch.tensor(seq_start_end)
    
    obs_traj = obs_traj.to('cuda')
    obs_traj_rel = obs_traj_rel.to('cuda')
    
    return [obs_traj, obs_traj_rel, seq_start_end]
    
    



def loadModel():
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



def markarray_callback(markarray):
# 1. 获取obs_traj
    global sample_time
    global obs_traj, obs_traj_rel, seq_start_end
    
    
    sample_time = sample_time + 1
    if sample_time == 2:
        [obs_traj, obs_traj_rel, seq_start_end] = get_obs_traj(markarray)
        sample_time = 0

    # [obs_traj, obs_traj_rel, seq_start_end] = get_obs_traj(markarray)
        
# 2. 加载模型,得到模型输出
    
    generator = loadModel()
    
    
    # pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    if obs_traj is not None:
        global pred_traj_fake_rel
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    else:
        rospy.loginfo("obs_traj is None.")
# 3. 发布模型输出
    
    # 将模型输出转换为绝对坐标
    
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    # print(pred_traj_fake.shape)
    
    # 发布模型输出
    print("pred_traj_fake.shape", pred_traj_fake.shape)
    # exit()
    trejector.publish_marker_array(pred_traj_fake.detach().cpu().numpy())

    






# # 加载预训练的PyTorch模型
# model_path = '/home/work_space/src/traj_pred/models/checkpoint/sgan-p-models/eth_8_model.pt'
# checkpoint = torch.load(model_path)
# args = AttrDict(checkpoint['args'])
# generator = TrajectoryGenerator(
#         obs_len=args.obs_len,
#         pred_len=args.pred_len,
#         embedding_dim=args.embedding_dim,
#         encoder_h_dim=args.encoder_h_dim_g,
#         decoder_h_dim=args.decoder_h_dim_g,
#         mlp_dim=args.mlp_dim,
#         num_layers=args.num_layers,
#         noise_dim=args.noise_dim,
#         noise_type=args.noise_type,
#         noise_mix_type=args.noise_mix_type,
#         pooling_type=args.pooling_type,
#         pool_every_timestep=args.pool_every_timestep,
#         dropout=args.dropout,
#         bottleneck_dim=args.bottleneck_dim,
#         neighborhood_size=args.neighborhood_size,
#         grid_size=args.grid_size,
#         batch_norm=args.batch_norm)
# generator.load_state_dict(checkpoint['g_state'])
# # if torch.cuda.is_available():
# generator.cuda()
# generator.train()



# # 定义模型输入
# obs_traj = np.array([[[11.7400,  6.5700], [12.2000,  4.9900]],

#                      [[11.0100,  6.5500], [11.5000,  4.7800]],

#                      [[10.2200,  6.5500], [10.8200,  4.4900]],

#                      [[ 9.3500,  6.5600], [10.3200,  4.2400]],

#                      [[ 8.5200,  6.5400], [ 9.7400,  4.0700]],

#                      [[ 7.6800,  6.5200], [ 9.1800,  3.8900]],

#                      [[ 6.8900,  6.3900], [ 8.4100,  3.7300]],

#                      [[ 6.0600,  6.3900], [ 7.8000,  3.6200]]])
# obs_traj_rel = np.array([[[ 0.0000,  0.0000], [ 0.0000,  0.0000]],

#                          [[-0.7300, -0.0200], [-0.7000, -0.2100]],

#                          [[-0.7900,  0.0000], [-0.6800, -0.2900]],

#                          [[-0.8700,  0.0100], [-0.5000, -0.2500]],

#                          [[-0.8300, -0.0200], [-0.5800, -0.1700]],

#                          [[-0.8400, -0.0200], [-0.5600, -0.1800]],

#                          [[-0.7900, -0.1300], [-0.7700, -0.1600]],

#                          [[-0.8300,  0.0000], [-0.6100, -0.1100]]])
# seq_start_end = [(0, 2)]

# obs_traj = torch.from_numpy(obs_traj).type(torch.float)
# obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
# seq_start_end = torch.tensor(seq_start_end).type(torch.long)

# obs_traj = obs_traj.to('cuda')
# obs_traj_rel = obs_traj_rel.to('cuda')

# # 得到模型输出
# start = time.time()
# pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
# end = time.time()

# print('time: ', end - start)



if __name__ == '__main__':
    # test_relative_to_abs()

    rospy.init_node('trajectory_Prediction_Node', anonymous=True, log_level=rospy.DEBUG)
    
    rospy.loginfo("trajectory_Prediction_Node started")

    rospy.Subscriber('mot_tracking/box', MarkerArray, markarray_callback)
    
    pub = rospy.Publisher('traj_pred/marker_array', MarkerArray, queue_size=8)


    rospy.spin()    
