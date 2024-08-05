import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import res18
from subproc_vec_env import SubprocVecEnv
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from utils.tool import log_and_print,DataProcessor
from buffer import ReplayBuffer


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
    capacity: int = 1e5
    train_ratio: float = 0.9
    
    
    # Actor
    actor_lr: float = 6e-4
    entropy: float = 0.01
    eps: float = 1e-5
    ppg: bool = False
    threshold: float = 0.5
    weight_decay: float = 0.01
    
    # Critic
    critic_lr: float = 1e-4
    
    # PPO
    clip: float = 0.20
    ppo_update: int = 25
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    num_epch_train: int = 600
    
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
            self.teachers = list(range(4))
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
                
                
           
def train_offline(RL_agent, eval_env, replaybuffer, hp):    
    
    file_time = hp.file_time
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-res18-{hp.total_timesteps}')
    process = DataProcessor(f'./output/{hp.file_name}/{file_time}/learner_')
    
    texts = []
    log_and_print(texts, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{hp.file_name}/{file_time}/models/"):
        output_directory = f'./output/{hp.file_name}/{file_time}/models/'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    if hp.checkpoint and not os.path.exists(f"./output/{hp.file_name}/{file_time}/checkpoint/models"):
        os.makedirs(f"./output/{hp.file_name}/{file_time}/checkpoint/models")
    
    
    ## 初次读取数据集加载环境
    teachers = hp.teachers
    s = teachers[0]
    world = s // 4 + 1
    stage = s % 4 + 1
    env_name = f"SuperMarioBros-{world}-{stage}-v0"
    eval_env = ProcessEnv(gym_super_mario_bros.make(env_name,render_mode='human', apply_api_compatibility=True))
    eval_env = SubprocVecEnv(hp.num_processes,env_name) if hp.num_processes > 1 else eval_env
    file_path = f'./dataset/{env_name}_data.h5'
    replaybuffer.load_data_from_hdf5(file_path)
    count = 0
    start_time = time.time()
    
    train_iterator = replaybuffer.get_iterator(is_train=True)
    val_iterator = replaybuffer.get_iterator(is_train=False)
    
    for t in range(int(hp.total_timesteps)):
        ## train
        s = time.time()
        total_train_loss = np.zeros(3, dtype=np.float64)
        for batch in train_iterator:
            actor_loss,value_loss,acc = RL_agent.learn(batch, writer)
            total_train_loss += np.array([actor_loss, value_loss,acc])
            
        avg_train_loss = total_train_loss / len(train_iterator)
        log_and_print(texts,f"Epoch {t}, Average Train Loss: {avg_train_loss[:2]} Average acc: {avg_train_loss[-1]*100:.2f}%")
        e = time.time()
        train_time = (e - s)
        
        # val
        total_val_loss = np.zeros(3, dtype=np.float64)
        for batch in val_iterator:
            actor_loss,value_loss,acc = RL_agent.evaluate(batch)
            total_val_loss += np.array([actor_loss, value_loss,acc])
        
        avg_val_loss = total_val_loss / len(val_iterator)
        log_and_print(texts,f"Epoch {t}, Average Validation Loss: {avg_val_loss[:2]} Average acc: {avg_val_loss[-1]*100:.2f}%")
        
        
        # eval
        s = time.time()
        text,total_reward = maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp)
        texts.append(text)
        if RL_agent.Maxscore[0]>70:
            eval_env.close()
            count += 1 
            RL_agent.Maxscore = (0.0,0)
            s = teachers[count]
            world = s // 4 + 1
            stage = s % 4 + 1
            env_name = f"SuperMarioBros-{world}-{stage}-v0"
            file_path = f'./dataset/{env_name}_data.h5'
            replaybuffer.load_data_from_hdf5(file_path)
            eval_env = ProcessEnv(gym_super_mario_bros.make(env_name,render_mode='human', apply_api_compatibility=True))
            eval_env = SubprocVecEnv(hp.num_processes,env_name) if hp.num_processes > 1 else eval_env
            train_iterator = replaybuffer.get_iterator(is_train=True)
            val_iterator = replaybuffer.get_iterator(is_train=False)
            
        e = time.time()
        eval_time = (e - s)
        process.process_input(total_reward,'return','eval/')
        process.process_input(train_time,'train_time','time/')
        process.process_input(eval_time,'eval_time','time/')
        process.process_input(count,'num','stage/')
        process.write_to_excel()
        RL_agent.save(f"./output/{hp.file_name}/{hp.file_time}/models/") 
        texts = [str(text) for text in texts]
        np.savetxt(f"./output/{hp.file_name}/{hp.file_time}/train.txt", texts, fmt='%s')
        
    eval_env.close()
        
            

                
def toTest(RL_agent, eval_env,file_name, hp):
    RL_agent.load(file_name)
    start_time = time.time()
    file_time = None
    for t in range(hp.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp)
    eval_env.close()
                
                
                
                
def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp, d4rl=False):
    
    text = []
    if hp.checkpoint or hp.test:
        log_and_print(text, "---------------------------------------")
        log_and_print(text, f"Evaluation at {t} time steps")
        log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")
        
        total_reward = []
        state = eval_env.reset()
        # state = state[0]
        episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        fin = 0
        
        while True:
            action,_,_ = RL_agent.select_action(np.array(state))
            next_state, reward, done, info = eval_env.step(action)
            # action = action[0]
            # next_state, reward, done, _, info = eval_env.step(action)
            state = next_state
            episode_rewards += reward
            mask =  1. - done.astype(np.float32)
            # mask = 1. - float(done)
            final_rewards *= mask
            final_rewards += (1. - mask) * episode_rewards
            episode_rewards *= mask
            if np.any(done==1):
                total_reward.extend(final_rewards[final_rewards!=0])
                final_rewards[final_rewards!=0] = 0
                for i, fo in enumerate(info):
                    if done[i] == 1:
                        if fo['flag_get']:
                            fin += 1
                # if info['flag_get']:
                #     fin += 1
                # state = eval_env.reset()
                # state = state[0]
                
            if len(total_reward)>=hp.eval_eps:
                break
        log_and_print(text, f"level cleared: {fin/hp.eval_eps*100:.2f}% , Average total reward over {hp.eval_eps} episodes: {np.mean(total_reward):.3f}")
        if d4rl:
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            log_and_print(text, f"D4RL score: {total_reward.mean():.3f}")
            
        if hp.checkpoint and not hp.test:
            if fin > hp.eval_eps:
                fin = hp.eval_eps
            fin = fin/hp.eval_eps*100
            score = np.mean(total_reward) + np.min(total_reward) + np.max(total_reward) + np.median(total_reward) - np.std(total_reward)
            flag = RL_agent.IsCheckpoint((fin,score))
            log_and_print(text, f"This Score：{(fin,score)} Max Score:{RL_agent.Maxscore}")
            if flag:
                RL_agent.save(f"./output/{hp.file_name}/{file_time}/checkpoint/models/")
        log_and_print(text, "---------------------------------------")
        return text,total_reward
                
                
                
                
    
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=3, type=int)
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


    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups

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
    hp.eval_eps = hp.eval_eps * hp.num_processes
    
    
    if gym.__version__ < '0.26':
        eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
    else:
        eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        
    
    eval_env = ProcessEnv(eval_env)
    
    
    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n
    
    replaybuffer = ReplayBuffer(hp.capacity,hp.discount,hp.train_ratio,hp.batch_size)
    
    RL_agent = res18.agent(state_dim,action_dim,hp)
    
    
    if args.test:
        file_name = f"./test/{args.file_name}/model/"
        toTest(RL_agent, eval_env,file_name, hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_offline(RL_agent, eval_env, replaybuffer, hp)
    

