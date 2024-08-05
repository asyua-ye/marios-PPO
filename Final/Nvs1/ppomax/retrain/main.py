import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import PPO
from utils.tool import DataProcessor,log_and_print
from buffer import RolloutBuffer
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from subproc_vec_env import SubprocVecEnv


@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 1
    discount: float = 0.9
    gae: float = 1.0
    grad: float = 0.5
    num_processes: int = 4
    num_steps: int = 2560
    device: torch.device = None
    max_steps: int = 0
    
    # Actor
    actor_lr: float = 1e-6
    entropy: float = 0.01
    log_std_max: int = 2
    log_std_min: int = -20
    eps: float = 1e-5
    std: bool = False
    ppg: bool = False
    share: bool = True
    threshold: float = 0.5
    warmup: float = 0.1
    
    # Critic
    critic_lr: float = 1e-4
    
    # PPO
    clip: float = 0.20
    ppo_update: int = 25
    mini_batch: int = 40
    value: float = 0.5
    actor: float = 1.0
    
    # RL
    env: str = "SuperMarioBros-1-1-v0"
    seed: int = 0
    test: bool = False

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 10
    max_timesteps: int = 400
    total_timesteps: int = 0
    eval: int = 1

    # File
    file_name: str = None
    
    # retrain 
    retrain: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def save_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for key, value in asdict(self).items():
                f.write(f"{key}: {value}\n")
    



def train_online(RL_agent, env, eval_env, rollout, hp):
    """
    不考虑单进程的情况了
    
    
    """
    state = env.reset()
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    rds = 0
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./output/{hp.file_name}/{file_time}/{file_time}-PPO-{hp.total_timesteps}')
    if hp.checkpoint and not os.path.exists(f"./output/{hp.file_name}/{file_time}/checkpoint/models"):
        os.makedirs(f"./output/{hp.file_name}/{file_time}/checkpoint/models")
    train = []
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./output/{hp.file_name}/{file_time}/models"):
        output_directory = f'./output/{hp.file_name}/{file_time}/models'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(f'./output/{hp.file_name}/{file_time}/models', 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{hp.file_name}/{file_time}/')
    Maxscore = [0.0,0.0]
    eval_fin = 0
    total_eval_reward = 0
    eval_all = 0
    for t in range(int(hp.total_timesteps+1)):
        s1 = time.time()
        
        total_reward = []
        fin = 0   
        for step in range(hp.num_steps):
            rounds += 1
            action,logit,value = RL_agent.select_action(np.array(state))
            next_state, reward, ep_finished, info = env.step(action)
            
            episode_rewards += reward
            mask =  1. - ep_finished.astype(np.float32)
            final_rewards *= mask
            final_rewards += (1. - mask) * episode_rewards
            temp = (1. - mask) * episode_rewards
            total_reward.extend(temp[temp!=0])
            episode_rewards *= mask
            rds += np.sum(ep_finished==1)
            
            
            reward = torch.from_numpy(reward.astype(np.float32)).to(hp.device)
            mask = torch.from_numpy(mask).to(hp.device).view(-1, 1)
            state = torch.from_numpy(state.astype(np.float64)).to(hp.device)
            action = torch.from_numpy(action.astype(np.float64)).to(hp.device)
            logit = torch.from_numpy(logit).to(hp.device)
            value = torch.from_numpy(value).to(hp.device)
                       
            rollout.insert(state, action,logit, value, reward, mask)
            state = next_state
            if torch.any(mask == 0).item() and np.any(final_rewards != 0):
                non_zero_rewards = final_rewards[final_rewards != 0]
                log_and_print(train, (
                    f"T: {t} Total T: {np.sum(rounds)} Total R: {rds}  mean: {np.mean(non_zero_rewards):.3f} "
                    f"mid: {np.median(non_zero_rewards):.3f} max: {np.max(non_zero_rewards):.3f} "
                    f"min: {np.min(non_zero_rewards):.3f}"
                        ))
                for i, fo in enumerate(info):
                        if mask[i] == 0:
                            if fo['flag_get']:
                                fin += 1     
        if hp.checkpoint:
            rs = len(total_reward)
            if fin > rs:
                fin = rs
            if rs!=0:
                all_fin = (fin + eval_fin) / (rs + eval_all) * 100
                sample_fin = fin / rs * 100
                all_score = np.min(total_reward) + np.min(total_eval_reward)
                sample_score = np.min(total_reward)
            else:
                all_fin = 0
                all_score = 0.0
            
            log_and_print(train, f"total {rs}  This Score：{(sample_fin,sample_score)}")
            log_and_print(train, f"checkpoint  total {rs + eval_all}  This Score：{(all_fin,all_score)} Max Score:{Maxscore}")
            flag = RL_agent.IsCheckpoint((all_fin,all_score),Maxscore)
            if flag:
                RL_agent.save(f"./output/{hp.file_name}/{file_time}/checkpoint/models/")
        e = time.time()
        sample_time = (e-s1)
        log_and_print(train, f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
        s = time.time()
        next_value = RL_agent.get_value(np.array(next_state))
        next_value = torch.from_numpy(next_value).view(-1,1).to(hp.device)
        states,actions,action_log_probs,advs,returns = rollout.computeReturn(next_value,mask)
        data=(np.copy(states), np.copy(actions),np.copy(action_log_probs), np.copy(advs),np.copy(returns))
        RL_agent.replaybuffer.push(data)
        log_and_print(train, f"T: {t}  R: {rds}   sample end begintrain！！")
        RL_agent.train(process,writer)
        e = time.time()
        train_time = (e-s)   
        if hp.checkpoint:
            s = time.time()
            text,total_eval_reward,eval_fin = maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp)
            e = time.time()
            eval_time = (e-s)
            train.extend(text)
            process.process_input(total_eval_reward,'Returns','eval/')
            eval_all = hp.eval_eps
        np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
        RL_agent.save(f'./output/{hp.file_name}/{file_time}/models/')
        process.process_input(sample_time,'sample_time(s)','time/')
        process.process_input(train_time,'train_time','time/')
        process.process_input(eval_time,'eval_time(s)','time/')
        e = time.time()
        total_time = (e-s1)
        process.process_input(total_time,'total_time(s)','time/')
        process.process_input(t,'Epoch')
        process.write_to_excel()
        
    env.close()
    eval_env.close()
    file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_and_print(train, f"\nend time: {file_time1}")
    np.savetxt(f"./output/{hp.file_name}/{file_time}/train.txt", train, fmt='%s')
            
            
def toTest(RL_agent, env, eval_env,file_name, hp):
    RL_agent.load(file_name)
    start_time = time.time()
    file_time = None
    for t in range(hp.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp)
    eval_env.close()
    
    
            
def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, hp):
    
    text = []
    if RL_agent.Train:
        RL_agent.training(False)
    if hp.checkpoint or hp.test:
        log_and_print(text, "---------------------------------------")
        log_and_print(text, f"Evaluation at {t} time steps")
        log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")
        
        total_reward = []
        state = eval_env.reset()
        episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
        fin = 0
        
        while True:
            action,_,_ = RL_agent.select_action(np.array(state))
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            episode_rewards += reward
            mask =  1. - done.astype(np.float32)
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
                
            if len(total_reward)>=hp.eval_eps:
                break
        log_and_print(text, f"level cleared: {fin/hp.eval_eps*100:.2f}% , Average total reward over {hp.eval_eps} episodes: {np.mean(total_reward):.3f}")
        log_and_print(text, "---------------------------------------")
        return text,total_reward,fin



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-8-4-v0", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=True, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=4, type=int)
    parser.add_argument("--max_timesteps", default=400, type=int)
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
        
        
    
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(args.env, new_step_api=True)
        eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)
    
    

    print("---------------------------------------")
    print(f"Algorithm: PPO, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")




    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU setups

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    hp = Hyperparameters(
        env=args.env,
        seed=args.seed,
        test=args.test,
        checkpoint=args.checkpoint,
        eval_eps=args.eval_eps,
        eval=args.eval,
        total_timesteps=args.max_timesteps,
        file_name=args.file_name,
        retrain = args.retrain
    )
    if hp.max_timesteps > args.max_timesteps:
        hp.max_timesteps = args.max_timesteps
        
    hp.max_steps = hp.max_timesteps * hp.ppo_update * hp.mini_batch
    hp.eval_eps = args.eval_eps * hp.num_processes
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    
    if not hp.test:
        envs = SubprocVecEnv(hp.num_processes,hp) if hp.num_processes > 1 else env
        
        
    eval_envs = SubprocVecEnv(hp.num_processes,hp) if hp.num_processes > 1 else eval_env
        
    RL_agent = PPO.agent(state_dim, action_dim, hp)
    
    rollout = RolloutBuffer(hp.num_steps, hp.num_processes, state_dim, action_dim, hp.gae, hp.discount)
        

    if args.test:
        file_name = f"./test/{args.file_name}/model/"
        toTest(RL_agent, env, eval_envs,file_name, hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        if args.retrain:
            file_name = f"./test/{args.file_name}/model/"
            RL_agent.load(file_name)
        train_online(RL_agent, envs, eval_envs, rollout, hp)







