#!/usr/bin/env python3
"""检查t+1时刻的图像来源（samples还是sweeps）"""

from nuscenes.nuscenes import NuScenes

# 初始化NuScenes数据集
nusc = NuScenes(version='v1.0-trainval',
                dataroot='/data/lfliang/project/DrivingForward/input_data/nuscenes',
                verbose=False)

# 读取训练集的sample列表
with open('dataset/nuscenes/train.txt', 'r') as f:
    filenames = f.readlines()

# 检查前10个samples
samples_count = 0
sweeps_count = 0

for i in range(min(10, len(filenames))):
    frame_idx = filenames[i].strip().split()[0]
    sample = nusc.get('sample', frame_idx)

    # 获取CAM_FRONT的sample_data
    cam_sample = nusc.get('sample_data', sample['data']['CAM_FRONT'])

    # 获取t+1时刻的sample_data
    if cam_sample['next']:
        fwd_sample = nusc.get('sample_data', cam_sample['next'])

        print(f"\nSample {i}:")
        print(f"  t时刻路径: {cam_sample['filename']}")
        print(f"  t+1时刻路径: {fwd_sample['filename']}")

        # 统计来源
        if fwd_sample['filename'].startswith('samples/'):
            samples_count += 1
        elif fwd_sample['filename'].startswith('sweeps/'):
            sweeps_count += 1

print(f"\n统计结果（前10个samples）:")
print(f"  t+1来自samples: {samples_count}")
print(f"  t+1来自sweeps: {sweeps_count}")
