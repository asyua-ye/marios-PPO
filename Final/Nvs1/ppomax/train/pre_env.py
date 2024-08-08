import numpy as np
import gymnasium as gym
import cv2
cv2.ocl.setUseOpenCL(False)
from gymnasium.spaces import Box
from utils.joypad_space import myJoypadSpace
from gymnasium.wrappers import FrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT,RIGHT_ONLY




MYACTION = [
    ['NOOP'],
    ['right'],
    ['left'],
    ['down'],
    ['up'],
    ['A'],
    ['B']
]



def Reward(info,reward,old_info,done,eight):
    
    p = 0
    h = 0
    x = 0
    # 计算累计score
    s = (info['score'] - old_info['score'])/100
    
    
    
    #不动惩罚
    x = (info['x_pos'] - old_info['x_pos'])
    if x==0:
        x = -1
    else:
        x = 0
    
    # 是否到达终点
    Flag = info['flag_get']
    if Flag:
        h = info['score'] + 50
    
    # 是否死亡
    if done and not Flag:
        p = -50
    
    
    # 直接指导
    new_done = False
    if info["world"] == 8 and info["stage"] == 4:
        
        if (eight == 1):
            reward = 0
            time = info["time"] - old_info["time"]
            reward += time
        
        if (eight == 3):
            reward = 0
            time = info["time"] - old_info["time"]
            reward += time
            y = (info["y_pos"] - old_info["y_pos"])
            y = np.clip(y, -1, 1)
            reward += y
        
        if (eight == 5):
            reward = 0
            time = info["time"] - old_info["time"]
            reward += time
        
        if (eight==0 and (np.abs(info["x_pos"]-old_info["x_pos"])>100)):
            p = - 200
            new_done = True
        elif (info["x_pos"] >=1250 and eight==0):
            eight += 1
        elif (eight == 1 and (info["x_pos"]>375) and info["x_pos"]<=1250):
            eight += 1
            p = - 200
            new_done = True
        elif (eight==1 and (info["x_pos"]>1840)):
            eight += 1
            s += 200
        elif (eight==2 and (np.abs(info["x_pos"]-old_info["x_pos"])>100)):
            p = - 200
            new_done = True               
        elif (eight==2 and (info["x_pos"]>2300)):
            eight += 1
        elif (eight==3 and (np.abs(info["x_pos"]-old_info["x_pos"])>100)):
            eight += 1
            s += 200    
        elif (eight==3 and (info["x_pos"]>2500)):
            eight += 1
            p = - 200
            new_done = True
        elif (eight==4 and (np.abs(info["x_pos"]-old_info["x_pos"])>100)):
            p = - 200
            new_done = True 
        elif (eight==4 and (info["x_pos"]>3560)):
            eight +=1
        elif (eight==5 and (np.abs(info["x_pos"]-old_info["x_pos"])>100)):
            eight += 1
            s += 200
        elif (eight==5 and (info["x_pos"]>3700)):
            p = - 200
            new_done = True
            
        
    
    if info["world"] == 7 and info["stage"] == 4:
        # print(f'x:{info["x_pos"]},old_x:{old_info["x_pos"]}')
        if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or (info["x_pos"] < old_info["x_pos"] - 500) or (
                        np.abs(info["x_pos"]-old_info["x_pos"])>100):
            p = -info['score']/10 - 200
            new_done = True
    if info["world"] == 4 and info["stage"] == 4:
        if   (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127) or (
                        np.abs(info["x_pos"]-old_info["x_pos"])>100):
            p = -info['score']/10 - 200
            new_done = True
            
            
    reward += (h + s + p + x)
        
    return reward,new_done,eight

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.old_info = None
        self.action_dim = env.action_space.n
        self.eight = 0

    def step(self, action):
        """
        action.shape:[numprocess,actiondim]
        many-hot:[0,0,1,0,1,0]
        """
        # total_reward = np.zeros(self.action_dim)
        total_reward = 0.0
        a = np.nonzero(action)[0]
        if len(a)!=0:
            real_a = [MYACTION[i] for i in a]
        else:
            real_a = [['NOOP']]
        
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(real_a)
            
            if self.old_info is None:
                self.old_info = info
            reward,new_done,self.eight = Reward(info,reward,self.old_info,done,self.eight)
            if new_done:
                done = new_done
            self.old_info = info
            
            # total_reward[a] += reward
            total_reward += reward
            if done:
                self.old_info = None
                self.eight = 0
                break
        return obs, total_reward, done, trunk, info
    
    def reset(self, **kwargs):
        
        self.old_info = None
        self.eight = 0
        
        return self.env.reset(**kwargs)


    
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        
        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
            
        
        return obs
    
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    
    
def ProcessEnv(env):
    env = myJoypadSpace(env, MYACTION)
    env = SkipFrame(env, skip=4)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env
    
    
    

