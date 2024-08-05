import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import PPO
import os
import time
from pre_env import ProcessEnv
from main import Hyperparameters
from utils.tool import log_and_print
import h5py
from tqdm import tqdm

# 创建文件夹
dataset_dir = './dataset/'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)




dtype = [('coins', int), ('flag_get', bool), ('life', int), ('score', int),
         ('stage', int), ('status', 'S10'), ('time', int), ('world', int),
         ('x_pos', int), ('y_pos', int)]


begin_world = 1
end_world = 1
begin_stage = 3
end_stage = 4
sampletime = 200
texts = []


for world in range(begin_world, end_world+1):
    for stage in range(begin_stage, end_stage+1):
        env_name = f"SuperMarioBros-{world}-{stage}-v0"
        
        # 创建环境
        env = gym_super_mario_bros.make(env_name, render_mode='rgb_array', apply_api_compatibility=True)
        env = ProcessEnv(env)
        
        # 读取agent
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
        hp = Hyperparameters()
        RL_agent = PPO.agent(state_dim, action_dim, hp)
        file_name = f"./test/{env_name}/model/"
        RL_agent.load(file_name)
        
        # 初始化数据存储
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_infos = []
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        # 运行环境
        state = env.reset()
        state = state[0]  # 这里根据环境的返回值格式可能需要调整
        episode_reward = 0
        episode_length = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=sampletime, desc=f"采集 {env_name} 数据")
        
        for i in range(sampletime):
            while True:
                action, _, _ = RL_agent.select_action(np.array(state))
                action = action[0]
                next_state, reward, done, _, info = env.step(action)
                
                
                # 存储数据
                all_states.append(state)
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done)
                all_infos.append(info)
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    if info['flag_get']:
                        success_count += 1
                    
                    state = env.reset()
                    state = state[0]
                    episode_reward = 0
                    episode_length = 0
                    break
            
            pbar.update(1)
        
        pbar.close()
        
        # 计算统计信息
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / sampletime
        
        structured_infos = np.array([(info['coins'], info['flag_get'], info['life'], info['score'],
                              info['stage'], info['status'].encode('ascii'), info['time'], info['world'],
                              info['x_pos'], info['y_pos']) for info in all_infos], dtype=dtype)
        
        
        
        # 创建HDF5文件
        h5_filename = os.path.join(dataset_dir, f"{env_name}_data.h5")
        with h5py.File(h5_filename, 'w') as f:
            # 存储原始数据
            f.create_dataset('states', data=np.array(all_states))
            f.create_dataset('actions', data=np.array(all_actions))
            f.create_dataset('rewards', data=np.array(all_rewards))
            f.create_dataset('dones', data=np.array(all_dones))
            f.create_dataset('infos', data=np.array(structured_infos))
            
            # 存储统计信息
            stats = f.create_group('statistics')
            stats.create_dataset('episode_rewards', data=np.array(episode_rewards))
            stats.create_dataset('episode_lengths', data=np.array(episode_lengths))
            stats.attrs['average_reward'] = avg_reward
            stats.attrs['average_length'] = avg_length
            stats.attrs['success_rate'] = success_rate
            stats.attrs['total_episodes'] = sampletime
        
        
        log_and_print(texts,f"数据已保存到 {h5_filename}")
        log_and_print(texts,f"平均奖励: {avg_reward:.2f}")
        log_and_print(texts,f"平均步长: {avg_length:.2f}")
        log_and_print(texts,f"通过率: {success_rate:.2%}")

log_and_print(texts,"所有数据采集完成")
np.savetxt(f"{dataset_dir}sample.txt", texts, fmt='%s')
        
        




