import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import SAC
from utils.tool import DataProcessor,log_and_print
from buffer import RolloutBuffer
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from subproc_vec_env import SubprocVecEnv


"""
多进程版本，同时给所有强化学习任务提供了个框架

在windows上，使用spawn，需要在每个子进程中单独初始化
env = gym_super_mario_bros.make(args.env, render_mode='human', apply_api_compatibility=True)
在每个子进程单独初始化了，虽然很丑，但是可以跑了



在linux，使用fork，方便一些，子进程可以直接在父进程中读取NES模拟器的内容
但是在实际运行中，会有很多奇怪的问题


也就是说，我想在windows中调试，需要改下代码，但是在linux中不需要

最终还是选择的spawn，主要fork会出现很多奇怪的问题，spawn只需要在子进程创建，然后就没问题了

同样的超参数，在linux和windows上居然训练的结果完全不一样...

"""

@dataclass
class Hyperparameters:
    # Generic
    buffer_size: int = 200
    discount: float = 0.6
    gae: float = 0.2
    grad: float = 0.5
    num_processes: int = 8
    num_steps: int = 600
    device: torch.device = None
    max_steps: int = 0
    
    # Actor
    actor_lr: float = 3e-4
    entropy: float = 0.01
    log_std_max: int = 2
    log_std_min: int = -20
    eps: float = 1e-5
    std: bool = False
    ppg: bool = True
    share: bool = True
    
    # Critic
    critic_lr: float = 3e-4
    
    # SAC
    batch: int = 256
    num_epch_train: int = int(3e2)
    adaptive_alpha: bool = False
    alpha: float = 0.2
    tau: float = 5e-3
    grad: float = 0.5
    
    # RL
    env: str = "SuperMarioBros-1-1-v0"
    seed: int = 0
    test: bool = False

    # Evaluation
    checkpoint: bool = True
    eval_eps: int = 2
    max_timesteps: int = 200
    eval: int = 1

    # File
    file_name: str = None

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
    state,mask = env.reset(),torch.ones((hp.num_processes,1)).to(hp.device)
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    start_time = time.time()
    file_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    writer = SummaryWriter(f'./{hp.file_name}/output/{file_time}/{file_time}-SAC-{hp.max_timesteps}')
    if hp.checkpoint and not os.path.exists(f"./{hp.file_name}/output/{file_time}/checkpoint/models"):
        os.makedirs(f"./{hp.file_name}/output/{file_time}/checkpoint/models")
    train = []
    log_and_print(train, (f"begin time:  {file_time}\n"))
    if not os.path.exists(f"./{hp.file_name}/output/{file_time}/models/"):
        output_directory = f'./{hp.file_name}/output/{file_time}/models/'
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    process = DataProcessor(f'./{hp.file_name}/output/{file_time}/')
    
    for t in range(int(hp.max_timesteps+1)):
        
        s1 = time.time()
        for step in range(hp.num_steps):
            rounds += 1
            action = RL_agent.select_action(np.array(state))
            next_state, reward, ep_finished, _ = env.step(action)
            if np.any(reward!=0):
                episode_rewards += np.max(reward[reward != 0], axis=-1)
            next_mask =  1. - ep_finished.astype(np.float32)
            final_rewards *= next_mask
            final_rewards += (1. - next_mask) * episode_rewards
            episode_rewards *= next_mask
            
            reward = torch.from_numpy(reward.astype(np.float32)).to(hp.device)
            next_mask = torch.from_numpy(next_mask).to(hp.device).view(-1, 1)
            state = torch.from_numpy(state.astype(np.float64)).to(hp.device)
            action = torch.from_numpy(action.astype(np.float64)).to(hp.device)
                       
            rollout.insert(state, action, reward, mask)
            state = next_state
            mask = next_mask
            if torch.any(mask == 0).item() and np.any(final_rewards != 0):
                non_zero_rewards = final_rewards[final_rewards != 0]
                log_and_print(train, (
                    f"T: {t} Total T: {np.sum(rounds)}  mean: {np.mean(non_zero_rewards):.3f} "
                    f"mid: {np.median(non_zero_rewards):.3f} max: {np.max(non_zero_rewards):.3f} "
                    f"min: {np.min(non_zero_rewards):.3f}"
                        ))
        e = time.time()
        sample_time = (e-s1)
        log_and_print(train, f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
        s = time.time()
        rollout.lastInsert(torch.from_numpy(next_state.astype(np.float64)).to(hp.device),next_mask)
        states,actions,next_states,rewards,masks,exps = rollout.computeReturn()
        data=(np.copy(states), np.copy(actions),np.copy(next_states), np.copy(rewards),np.copy(masks),np.copy(exps))
        RL_agent.replaybuffer.push(data)
        log_and_print(train, f"T: {t}   sample end begintrain！！")
        RL_agent.train(process,writer)
        e = time.time()
        train_time = (e-s)   
        evals = []
        if hp.checkpoint:
            s = time.time()
            text,total_reward = maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,file_time, hp)
            e = time.time()
            eval_time = (e-s)
            train.extend(text)
            process.process_input(total_reward,'Returns','eval/')
        np.savetxt(f"./{hp.file_name}/output/{file_time}/train.txt", train, fmt='%s')
        RL_agent.save(f'./{hp.file_name}/output/{file_time}/models/')
        process.process_input(sample_time,'sample_time(s)','time/')
        process.process_input(train_time,'train_time','time/')
        process.process_input(eval_time,'eval_time(s)','time/')
        e = time.time()
        total_time = (e-s1)
        process.process_input(total_time,'total_time(s)','time/')
        process.process_input(t,'Epoch')
        process.write_to_excel()
        
    env.close()
    file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_and_print(train, f"\nend time: {file_time1}")
    np.savetxt(f"./{hp.file_name}/output/{file_time}/train.txt", train, fmt='%s')
            
            
def toTest(RL_agent, env, eval_env,file_name, hp):
    RL_agent.load(f"./models/")
    evals = []
    start_time = time.time()
    file_time = None
    for t in range(hp.eval):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,file_time, hp)
    
    
            
def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,file_time, hp, d4rl=False):
    
    text = []
    if hp.checkpoint or hp.test:
        log_and_print(text, "---------------------------------------")
        log_and_print(text, f"Evaluation at {t} time steps")
        log_and_print(text, f"Total time passed: {round((time.time() - start_time) / 60., 2)} min(s)")

        total_reward = np.zeros(hp.eval_eps)
        for ep in range(hp.eval_eps):
            state, done = eval_env.reset(), False
            state = state[0]
            while not done:
                action = RL_agent.select_action(np.array(state))
                next_state, reward, done, _, _ = eval_env.step(action)
                total_reward[ep] += np.max(reward)
                state = next_state
                
        log_and_print(text, f"Average total reward over {hp.eval_eps} episodes: {total_reward.mean():.3f}")
        if d4rl:
            total_reward = eval_env.get_normalized_score(total_reward) * 100
            log_and_print(text, f"D4RL score: {total_reward.mean():.3f}")
        evals.append(total_reward)
        
        if hp.checkpoint and not hp.test:
            np.save(f"./{hp.file_name}/output/{file_time}/checkpoint/{hp.file_name}", evals)
            score = np.mean(total_reward) + np.min(total_reward) + np.max(total_reward) + np.median(total_reward) - np.std(total_reward)
            flag = RL_agent.IsCheckpoint(score)
            log_and_print(text, f"This Score：{score} Max Score:{RL_agent.Maxscore}")
            if flag:
                RL_agent.save(f"./{hp.file_name}/output/{file_time}/checkpoint/models/")
        if hp.test:
            np.save(f"./{hp.file_name}/output/{file_time}/evals", evals)
        log_and_print(text, "---------------------------------------")
        return text,total_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)

    # Evaluation
    parser.add_argument("--checkpoint", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_eps", default=2, type=int)
    parser.add_argument("--max_timesteps", default=1000, type=int)
    parser.add_argument("--eval",default=1,type=int)
    # File
    parser.add_argument('--file_name', default=None)
    
    args = parser.parse_args()
    
    
    
    
    if args.file_name is None:
        args.file_name = f"{args.env}"

    if not os.path.exists(f"./{args.file_name}/output"):
        os.makedirs(f"./{args.file_name}/output")
        
        
    
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(args.env, new_step_api=True)
        eval_env = gym_super_mario_bros.make(args.env, new_step_api=True)
    else:
        env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        eval_env = gym_super_mario_bros.make(args.env,render_mode='human', apply_api_compatibility=True)
        
    
    env = ProcessEnv(env)
    eval_env = ProcessEnv(eval_env)
    
    

    print("---------------------------------------")
    print(f"Algorithm: SAC, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")





    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    hp = Hyperparameters(
        env=args.env,
        seed=args.seed,
        test=args.test,
        checkpoint=args.checkpoint,
        eval_eps=args.eval_eps,
        max_timesteps=args.max_timesteps,
        eval=args.eval,
        file_name=args.file_name
    )
    hp.max_steps = hp.max_timesteps * hp.num_epch_train
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    
    envs=SubprocVecEnv(hp.num_processes,hp) if hp.num_processes > 1 else env
    
    
    RL_agent = SAC.agent(state_dim, action_dim, max_action, hp)
    
    rollout = RolloutBuffer(hp.num_steps, hp.num_processes, state_dim, action_dim, hp.discount)
        
    torch.autograd.set_detect_anomaly(True)
    
    
    if args.test:
        file_name = f''
        toTest(RL_agent, env, eval_env,file_name, hp)
    else:
        from torch.utils.tensorboard import SummaryWriter
        train_online(RL_agent, envs, eval_env, rollout, hp)







