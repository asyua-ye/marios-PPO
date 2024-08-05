import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import PPO
from main import Hyperparameters
import os


begin_world = 1
end_world = 8
begin_stage = 1
end_stage = 4

for world in range(begin_world, end_world):
    for stage in range(begin_stage, end_stage):
        # 创建环境
        env_name = f"SuperMarioBros-{world}-{stage}-v0"
        env = gym_super_mario_bros.make(env_name, apply_api_compatibility=True)

        # 使用 Monitor 包装环境
        video_dir = f'./video/{env_name}'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = gym.wrappers.Monitor(env, video_dir, force=True)
        
        # 读取agent
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
        hp = Hyperparameters()
        RL_agent = PPO.agent(state_dim, action_dim, hp)
        file_name = f"./test/{env_name}/model/"
        RL_agent.load(file_name)
        
        # 运行环境
        state = env.reset()
        state = state[0]  # 这里根据环境的返回值格式可能需要调整
        
        while True:
            action, _, _ = RL_agent.select_action(np.array(state))
            action = action[0]
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            
            if done:
                break   

        env.close()