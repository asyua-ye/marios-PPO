import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import PPO
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from ActorLearner import ActorLearner,maybe_evaluate_and_print



"""
异步采样


一个经典的生产者消费者的问题，actor不停的生产，learner不停的消费
最理想的情况是，两个角色都自己干自己的，几乎不停。

actor,会描述自己的采样，每次读取新的模型参数(读取200次就结束采样)，有个提示
learner，就闷头训练就可以了

"""

@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 12
    discount: float = 0.9
    gae: float = 1.0
    grad: float = 0.5
    num_processes: int = 4
    num_steps: int = 300
    device: torch.device = None
    max_steps: int = 0
    tau: float = 5e-3
    
    # Actor
    actor_lr: float = 1e-4
    entropy: float = 0.01
    eps: float = 1e-5
    ppg: bool = False
    
    # Critic
    critic_lr: float = 1e-4
    
    # LAP
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.001
    epsilon: float = 0.01
    
    
    # PPO
    clip: float = 0.25
    ppo_update: int = 25
    mini_batch: int = 40
    c: float = 3.0
    num_epch_train: int = 1000
    batch_size: int = 256
    
    # RL
    env: str = "SuperMarioBros-1-1-v0"
    seed: int = 0
    test: bool = False

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 2
    max_timesteps: int = 200
    total_timesteps: int = 400
    eval: int = 1

    # File
    file_name: str = None
    file_time: datetime = None

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
    



def train_online(hp):
    """
    不考虑单进程的情况了
    
    
    """
    
    actor_learner = ActorLearner(hp) 
    
    actor_learner.start()
    actor_learner.join()
    
    
            
            
def toTest(RL_agent, env, eval_env,file_name, hp):
    RL_agent.load(f"./{hp.file_name}/output/{file_name}/checkpoint/models/")
    evals = []
    start_time = time.time()
    
    for t in range(hp.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, hp)
    
    
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-8-4-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=4, type=int)
    parser.add_argument("--max_timesteps", default=1000, type=int)
    parser.add_argument("--eval",default=1,type=int)
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    

    
        
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./output/{args.file_name}"):
        os.makedirs(f"./output/{args.file_name}")
        
        
    if not os.path.exists(f"./test/{args.file_name}/model"):
        os.makedirs(f"./test/{args.file_name}/model")
        
        
    

    print("---------------------------------------")
    print(f"Algorithm: PPO, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    hp = Hyperparameters(
        env=args.env,
        seed=args.seed,
        test=args.test,
        checkpoint=args.checkpoint,
        eval_eps=args.eval_eps,
        eval=args.eval,
        file_name=args.file_name,
        file_time = file_time
    )
    if hp.max_timesteps > args.max_timesteps:
        hp.max_timesteps = args.max_timesteps
    
    hp.max_steps = hp.max_timesteps * hp.ppo_update * hp.mini_batch
    hp.total_timesteps = args.max_timesteps
    
    if args.test:
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(args.env, new_step_api=True)
            eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
        else:
            env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
            eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
            
        
        env = ProcessEnv(env)
        eval_env = ProcessEnv(eval_env)
        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
        RL_agent = PPO.agent(state_dim, action_dim, hp)
        file_name = f''
        toTest(RL_agent, env, eval_env,file_name, args)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_online(hp)







