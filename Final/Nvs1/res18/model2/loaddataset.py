import h5py
import numpy as np

# 打开HDF5文件
file_path = './dataset/SuperMarioBros-1-1-v0_data.h5'  # 根据你的实际文件名和路径修改
with h5py.File(file_path, 'r') as f:
    # 列出所有的主键
    print("Keys: %s" % f.keys())
    
    # 访问数据集
    states = f['states'][:]
    actions = f['actions'][:]
    rewards = f['rewards'][:]
    dones = f['dones'][:]
    infos = f['infos'][:]
    
    # 访问统计信息
    stats = f['statistics']
    episode_rewards = stats['episode_rewards'][:]
    episode_lengths = stats['episode_lengths'][:]
    avg_reward = stats.attrs['average_reward']
    avg_length = stats.attrs['average_length']
    success_rate = stats.attrs['success_rate']
    total_episodes = stats.attrs['total_episodes']

    # 打印部分数据检查
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    print(f"Infos shape: {infos.shape}")
    
    # 打印统计信息
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode lengths: {episode_lengths}")
    print(f"Average reward: {avg_reward}")
    print(f"Average length: {avg_length}")
    print(f"Success rate: {success_rate}")
    print(f"Total episodes: {total_episodes}")

    # 检查部分数据
    print(f"Sample state: {states[0]}")
    print(f"Sample action: {actions[0]}")
    print(f"Sample reward: {rewards[0]}")
    print(f"Sample done: {dones[0]}")
    print(f"Sample info: {infos[0]}")