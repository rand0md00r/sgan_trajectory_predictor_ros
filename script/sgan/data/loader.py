from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,                                   # 一个数据集
        batch_size=args.batch_size,             # batch_size, 从args中获取
        shuffle=True,                           # 是否打乱数据集, 打乱的意思是每次取batch_size个数据的时候, 是随机取的
        num_workers=args.loader_num_workers,    # 多线程读取数据集, 从args中获取
        collate_fn=seq_collate)                 # 一个函数, 用于将数据集中的数据按照batch_size进行分组, 从args中获取
    return dset, loader
