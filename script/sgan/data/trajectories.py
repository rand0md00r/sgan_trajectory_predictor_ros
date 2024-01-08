import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim) # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            line = [float(i) for i in line]  # 将字符串转换为浮点数
            data.append(line)                # data是一个list，里面包含了每个行人的轨迹[index, ped_id, x, y]
    return np.asarray(data)

def visualize_trajectories(data):
    # 创建一个空的字典，用于存储每个行人的轨迹数据
    trajectories = {}

    # 创建一个matplotlib的动态图像
    fig = plt.figure()

    # 遍历数据，将每个行人的轨迹数据添加到字典中，如果一个行人的位置数据多于8个，丢掉最旧的数据
    for line in data:
        ped_id = line[1]
        if ped_id not in trajectories:
            trajectories[ped_id] = []
        trajectories[ped_id].append(line[2:])
        if len(trajectories[ped_id]) > 8:
            trajectories[ped_id] = trajectories[ped_id][-8:]

        # 清除当前的图像
        plt.clf()

        # 遍历字典，对每个行人的轨迹数据进行绘图，每个行人的轨迹使用不同的颜色
        for ped_id, trajectory in trajectories.items():
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1])

        # 更新图像
        plt.pause(0.01)

    plt.show()


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)      # traj是一个numpy数组，shape为(2, traj_len)
    - traj_len: Len of trajectory                   # traj_len是轨迹的长度
    - threshold: Minimum error to be considered for non linear traj # threshold是考虑为非线性轨迹的最小误差
    Output:
    - int: 1 -> Non Linear 0-> Linear               # int是1代表非线性轨迹，0代表线性轨迹
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    # polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False), deg是拟合多项式的次数，rcond是奇异值分解的阈值，full是返回值的类型
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]     
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format    # data_dir是包含数据集文件的目录
        <frame_id> <ped_id> <x> <y>                                     # frame_id是帧id，ped_id是行人id，x和y是坐标
        - obs_len: Number of time-steps in input trajectories           输入轨迹中的时间步数
        - pred_len: Number of time-steps in output trajectories         输出轨迹中的时间步数
        - skip: Number of frames to skip while making the dataset       在创建数据集时要跳过的帧数
        - threshold: Minimum error to be considered for non linear traj 考虑为非线性轨迹的最小误差
        when using a linear predictor  # 使用线性预测器时                                 
        - min_ped: Minimum number of pedestrians that should be in a seqeunce   应在序列中的最小行人数
        - delim: Delimiter in the dataset files                                 数据集文件中的分隔符
        """
        super(TrajectoryDataset, self).__init__()           # 调用父类的构造函数，初始化父类的属性，这里是空的

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)           # 返回指定的文件夹包含的文件或文件夹的名字的列表
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files] # 将文件夹名字与路径拼接起来
        num_peds_in_seq = []            # 每个序列中的行人数
        seq_list = []                   # 所有轨迹的列表
        seq_list_rel = []               # 所有轨迹相对坐标形式的列表
        loss_mask_list = []             # 损失掩码列表
        non_linear_ped = []             # 非线性行人列表
        for path in all_files:
            data = read_file(path, delim)
            # visualize_trajectories(data)      # 查看数据集中的轨迹
            frames = np.unique(data[:, 0]).tolist()     # 帧列表
            frame_data = []                             # 帧数据
            for frame in frames:
                # 将数据按帧分组，frame_data是一个list，里面包含了每一帧的数据，每一帧的数据是一个numpy数组
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            for idx in range(0, num_sequences * self.skip + 1, skip):# range(start, stop[, step])，step是步长
                # 当前序列数据是按帧将观测序列和预测序列拼接起来的，共有seq_len帧数据
                curr_seq_data = np.concatenate( # concatenate((a1, a2, ...), axis=0, out=None)，axis=0代表按行拼接
                    frame_data[idx: idx + self.seq_len], axis=0)   # 从idx到idx+self.seq_len的帧数据拼接起来
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 当前序列中的行人id,存入一个numpy数组，[1. 2. 3. 4. 5. 6. 7.]
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, # 当前序列的相对坐标形式，三维0数组，第一维是行人数，第二维是坐标数，第三维是序列长度
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len)) # 当前完整序列，三维0数组，第一维是行人数，第二维是坐标数，第三维是序列长度
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==     # 第t帧中行人id为ped_id的行人的数据
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)      # 将数据保留4位小数
                    # 计算在当前行人的序列前后需要填充的元素数量，以保证所有序列的长度一致。  重要！！！
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])        # 将坐标转置，shape为(seq_len, 2)
                    curr_ped_seq = curr_ped_seq # 奇妙的操作，不知道为什么要这么做，可能是为了将坐标转换为相对坐标，但是这里的坐标已经是相对坐标了，所以这里的操作没有意义
                    # Make coordinates relative # 将坐标转换为相对坐标          重要！！！
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq         # 当前完整序列，包含了所有行人的数据。一个序列是模型的输入
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq # 当前序列的相对坐标形式，包含了所有行人的数据
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])             
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])     # seq_list_rel是一个list，里面包含了每个序列的相对坐标

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
