import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import res18
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from ActorLearner import ActorLearner,maybe_evaluate_and_print


@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 5
    discount: float = 0.9
    gae: float = 1.0
    grad: float = 0.5
    num_processes: int = 4
    num_steps: int = 2560
    device: torch.device = None
    max_steps: int = 0
    batch_size: int = 128
    
    
    # Actor
    actor_lr: float = 1e-4
    entropy: float = 0.01
    log_std_max: int = 2
    log_std_min: int = -20
    eps: float = 1e-5
    std: bool = False
    ppg: bool = False
    share: bool = True
    threshold: float = 0.5
    
    # Critic
    critic_lr: float = 1e-4
    
    # PPO
    clip: float = 0.20
    ppo_update: int = 25
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    num_epch_train: int = 500
    
    # RL
    env: str = "SuperMarioBros-1-1-v0"
    seed: int = 0
    test: bool = False
    teachers: list = None

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 10
    max_timesteps: int = 800
    total_timesteps: int = 0
    eval: int = 1

    # File
    file_name: str = None
    file_time: datetime = None

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.teachers is None:
            self.teachers = list(range(2))
        
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
    
    
            
            
def toTest(RL_agent, eval_env,file_name, hp):
    RL_agent.load(f"./{hp.file_name}/output/{file_name}/checkpoint/models/")
    evals = []
    start_time = time.time()
    
    for t in range(hp.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, hp)
                
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=4, type=int)
    parser.add_argument("--max_timesteps", default=2000, type=int)
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
    print(f"Algorithm: res18, Env: {args.env}, Seed: {args.seed}")
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
        
    hp.max_steps = hp.max_timesteps * hp.num_epch_train
    hp.total_timesteps = args.max_timesteps
    hp.eval_eps = hp.num_processes * args.eval_eps
    
    if args.test:
        if gym.__version__ < '0.26':
            eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
        else:
            eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
            
        eval_env = ProcessEnv(eval_env)
        state_dim = eval_env.observation_space.shape
        action_dim = eval_env.action_space.n
        RL_agent = res18.agent(state_dim, action_dim, hp)
        file_name = f''
        toTest(RL_agent, eval_env,file_name, args)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_online(hp)
                
