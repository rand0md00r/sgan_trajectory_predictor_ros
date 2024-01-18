import pandas as pd

# 读取 txt 文件
input_txt_path = "/home/work_space/src/traj_pred/dataset/raw/seu1_raw.txt"
data = pd.read_csv(input_txt_path, sep=" ", header=None)
data.columns = ["frame_id", "pedestrian_id", "x", "y"]

# 将data中的每行中的每个元素都转换成浮点类型
data = data.astype(float)


# 计算训练集、验证集和测试集的大小
train_size = int(0.8 * len(data))


# 划分数据集
train_data = data[:train_size]
val_data = data[train_size:]


# 保存数据
output_folder_path = "/home/work_space/src/traj_pred/dataset/seu1"
train_data.to_csv('{}/train/train.txt'.format(output_folder_path), sep=' ', index=False, header=False)
val_data.to_csv('{}/val/val.txt'.format(output_folder_path), sep=' ', index=False, header=False)