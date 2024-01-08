import argparse
import os
import torch

from attrdict import AttrDict           # 一个字典，可以通过点操作符访问字典的key

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="models/sgan-models/eth_8_model.pt", type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
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
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():  # 不需要计算梯度，也不需要反向传播，节省内存，加快计算速度，只能用于推断
        # pytorch的Dataloader将数据集中的数据按照batch_size进行分组，每个batch_size的数据都是一个list
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]     # 将batch中的每个tensor放到GPU上
            # batch包含了：
            # 观测轨迹obs_traj
            # 预测轨迹真值pred_traj_gt
            # 相对观测轨迹obs_traj_rel
            # 相对预测轨迹真值pred_traj_gt_rel
            # 非线性行人non_linear_ped
            # 损失掩码loss_mask
            # 序列开始和结束seq_start_end
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):                    # num_samples代表采样次数
                # generator是一个TrajectoryGenerator类的实例， 其__call__方法返回pred_traj_fake_rel，
                # pred_traj_fake_rel是一个tensor，shape为(8, ???, 2), 
                # 8代表预测步长，???代表每个时间步长中的样本数量，2代表每个样本的x和y坐标
                # ???是什么呢？ 
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                # 将预测序列的相对坐标转换为绝对坐标，
                # pred_traj_fake是一个tensor，shape为(12, 8, 2)，
                # 代表预测的轨迹
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                # TODO: 预测轨迹的输出在这里，
                # 后续从这里开始写ROS节点，将预测轨迹的输出发送到ROS中！！！！！！

                
                
                
                
                
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    # 1.加载模型
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)       # 得到generator模型
        # checkoutpoint包含了：['g_state', 'args']
        _args = AttrDict(checkpoint['args'])        # 得到模型参数，用.访问key
        path = get_dset_path(_args.dataset_name, args.dset_type)
        # 2.加载数据集
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
