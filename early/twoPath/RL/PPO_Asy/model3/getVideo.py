import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import PPO
import os
import time
from moviepy.editor import ImageSequenceClip
from pre_env import ProcessEnv
from main import Hyperparameters
from utils.tool import log_and_print


begin_world = 1
end_world = 8
begin_stage = 1
end_stage = 4
        


test = []
for world in range(begin_world, end_world+1):
    for stage in range(begin_stage, end_stage+1):
        # 创建video
        env_name = f"SuperMarioBros-{world}-{stage}-v0"
        video_dir = f'./video/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        # 创建环境
        env = gym_super_mario_bros.make(env_name,render_mode='rgb_array', apply_api_compatibility=True)
        env = ProcessEnv(env)
        
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
        test_time = 10
        fin = 0
        i = 0
        saved_video = False  # 是否已保存成功视频
        frames = []
        while True:
            frame = np.copy(env.render())
            frames.append(frame)
            action, _, _ = RL_agent.select_action(np.array(state))
            action = action[0]
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            
            if done:
                if info['flag_get']:
                    fin += 1
                    if not saved_video:
                        clip = ImageSequenceClip(frames, fps=24)
                        clip.write_videofile(f"{video_dir}/{env_name}.mp4")
                        saved_video = True
                frames = []
                i += 1
                time.sleep(0.1)
                state = env.reset()
                state = state[0]
                if i >= test_time:  # 达到测试次数上限
                    break
            
        log_and_print(test,f"{env_name},通过率:{fin/test_time*100:.2f}%")
        env.close()
        np.savetxt(f"./video/test.txt", test, fmt='%s')