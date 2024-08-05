import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import PPO
import os
import time
from moviepy.editor import ImageSequenceClip,VideoFileClip
from pre_env import ProcessEnv
from main import Hyperparameters
from utils.tool import log_and_print


def mp4_to_gif(mp4_path, gif_path, fps=10, scale=1.0):
    """
    将MP4视频转换为GIF。
    
    :param mp4_path: MP4文件的路径
    :param gif_path: 输出GIF文件的路径
    :param fps: GIF的帧率，默认为10
    :param scale: GIF的缩放比例，默认为0.5（即原视频大小的一半）
    """
    # 读取视频文件
    video = VideoFileClip(mp4_path)
    
    # 调整视频大小
    video_resized = video.resize(scale)
    
    # 转换为GIF
    video_resized.write_gif(gif_path, fps=fps)
    
    # 关闭视频对象
    video.close()



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
        
        gif_dir = './gif/'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            
        gif_path = f"{gif_dir}/{env_name}.gif"
        mp4_path = f"{video_dir}/{env_name}.mp4"
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
        test_time = 200
        fin = 0
        i = 0
        saved_video = False  # 是否已保存成功视频
        frames = []
        
        
        while True:
            if not saved_video:
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
                        last_frame = frames[-1]
                        extra_frames = 20
                        frames.extend([last_frame] * extra_frames)
                        
                        clip = ImageSequenceClip(frames, fps=12)
                        clip.write_videofile(mp4_path)
                        saved_video = True
                        if os.path.exists(mp4_path):
                            mp4_to_gif(mp4_path, gif_path)
                            print(f"转换完成: {gif_path}")
                        
                frames = []
                i += 1
                state = env.reset()
                state = state[0]
                if i >= test_time:  # 达到测试次数上限
                    break
            
        log_and_print(test,f"{env_name},测试次数:{test_time},通过率:{fin/test_time*100:.2f}%")
        env.close()
        np.savetxt(f"./video/test.txt", test, fmt='%s')