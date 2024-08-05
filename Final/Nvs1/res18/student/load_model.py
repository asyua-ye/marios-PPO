import logging
import gym_super_mario_bros
import PPO as model1
from tqdm import tqdm
from pre_env import ProcessEnv

def getTeacher(stages,hp):
    """
    0-31
    0对应1-1
    31对应的是8-4
    """
    assert all(0 <= s <= 31 for s in stages), "stage里面的数只能在0-31之间"
    stages = sorted(stages)
    teachers = []

    for s in tqdm(stages, desc="Loading teachers"):
        world = s // 4 + 1
        stage = s % 4 + 1
        
        env_name = f"SuperMarioBros-{world}-{stage}-v0"
        
        try:
            # 创建环境
            with gym_super_mario_bros.make(env_name, render_mode='rgb_array', apply_api_compatibility=True) as env:
                env = ProcessEnv(env)
                
                # 读取agent
                state_dim = env.observation_space.shape
                action_dim = env.action_space.n
                RL_agent = model1.agent(state_dim, action_dim, hp)
                file_name = f"./test/{env_name}/model/"
                RL_agent.load(file_name)
                
                teachers.append(RL_agent)
                
            logging.info(f"Successfully loaded teacher for {env_name}")
        except Exception as e:
            logging.error(f"Error loading teacher for {env_name}: {str(e)}")

    logging.info(f"Loaded {len(teachers)} teachers out of {len(stages)} requested stages")
    return teachers