import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import argparse
import os
import time
import datetime
import PPO
import multiprocessing as mp
from utils.tool import log_and_print,DataProcessor
from buffer import RolloutBuffer,ReplayBuffer
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from subproc_vec_env import SubprocVecEnv
from pre_env import ProcessEnv
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.managers import BaseManager


def actor(hp, replaybuffer, event, counter):
    eval_env = gym_super_mario_bros.make(hp.env, render_mode='human', apply_api_compatibility=True)
    eval_env = ProcessEnv(eval_env)
    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    env = SubprocVecEnv(hp.num_processes, hp)
    RL_agent = PPO.agent(state_dim, action_dim, max_action, hp)
    rollout = RolloutBuffer(hp.num_steps, hp.num_processes, state_dim, action_dim, hp.gae, hp.discount)
    state = env.reset()
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    start_time = time.time()

    train = []
    log_and_print(train, (f"begin time:  {hp.file_time}\n"))
    if hp.checkpoint and not os.path.exists(f"./{hp.file_name}/output/{hp.file_time}/checkpoint/models"):
        os.makedirs(f"./{hp.file_name}/output/{hp.file_time}/checkpoint/models")

    output_directory = f"./{hp.file_name}/output/{hp.file_time}/models/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    process = DataProcessor(f'./{hp.file_name}/output/{hp.file_time}/actor_')

    t = 0
    while True:
        if t == 0 or event.is_set():

            if t == hp.max_timesteps:
                counter.value = -2
                env.close()
                file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                log_and_print(train, f"\nend time: {file_time1}")
                np.savetxt(f"./{hp.file_name}/output/{hp.file_time}/train.txt", train, fmt='%s')
                break
            
            eval_time = 0
            total_reward = np.zeros(hp.eval_eps)
            if t != 0:
                RL_agent.load(output_directory)
                evals = []
                if hp.checkpoint:
                    s = time.time()
                    text,total_reward = maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time,hp.file_time, hp)
                    e = time.time()
                    eval_time = (e-s)
                    train.extend(text)
                    
                
            t += 1
            s1 = time.time()
            for step in range(hp.num_steps):
                rounds += 1
                action, logit, value, z = RL_agent.select_action(np.array(state))
                next_state, reward, ep_finished, _ = env.step(action)
                if np.any(reward != 0):
                    episode_rewards += np.max(reward[reward != 0], axis=-1)
                mask = 1. - ep_finished.astype(np.float32)
                final_rewards *= mask
                final_rewards += (1. - mask) * episode_rewards
                episode_rewards *= mask

                reward = torch.from_numpy(reward.astype(np.float32)).to(hp.device)
                mask = torch.from_numpy(mask).to(hp.device).view(-1, 1)
                state = torch.from_numpy(state.astype(np.float64)).to(hp.device)
                action = torch.from_numpy(action.astype(np.float64)).to(hp.device)
                logit = torch.from_numpy(logit).to(hp.device)
                value = torch.from_numpy(value).to(hp.device)
                z = torch.from_numpy(z).to(hp.device)

                rollout.insert(state, action, logit, value, reward, mask, z)
                state = next_state
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
            next_value = RL_agent.get_value(np.array(next_state))
            next_value = torch.from_numpy(next_value).view(-1, 1).to(hp.device)
            states, actions, action_log_probs, advs, zs, returns = rollout.computeReturn(next_value, mask)
            data = (np.copy(states), np.copy(actions), np.copy(action_log_probs), np.copy(advs), np.copy(zs), np.copy(returns))
            replaybuffer.push(data)
            e = time.time()
            train_time = (e-s)   
            
            # 每次循环结束时保存训练日志
            np.savetxt(f"./{hp.file_name}/output/{hp.file_time}/train.txt", train, fmt='%s')
            process.process_input(total_reward,'Returns','eval/')
            process.process_input(sample_time,'sample_time(s)','time/')
            process.process_input(train_time,'train_time','time/')
            process.process_input(eval_time,'eval_time(s)','time/')
            e = time.time()
            total_time = (e-s1)
            process.process_input(total_time,'total_time(s)','time/')
            process.process_input(t,'Epoch')
            process.write_to_excel()
            
            with counter.get_lock():
                counter.value += 1
            event.clear()
        
        
        

def learner(hp, replaybuffer, event, counter):
    writer = SummaryWriter(f'./{hp.file_name}/output/{hp.file_time}/{hp.file_time}-PPO-{hp.max_timesteps}')
    process = DataProcessor(f'./{hp.file_name}/output/{hp.file_time}/learner_')
    eval_env = gym_super_mario_bros.make(hp.env)
    eval_env = ProcessEnv(eval_env)
    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RL_agent = PPO.agent(state_dim, action_dim, max_action, hp)

    try:
        while True:
            with counter.get_lock():
                count = counter.value

            if count == -2:
                break
            
            if count != -1:
                count %= hp.buffer_size
                
                for i in range(hp.ppo_update):
                    samples = replaybuffer.PPOsample([count], hp.mini_batch)
                    for sample in samples:
                        
                        state, action, action_log_prob, ret, adv, z = sample
                        sample = (torch.tensor(state).to(device),
                                torch.tensor(action).to(device),
                                torch.tensor(action_log_prob).to(device),
                                torch.tensor(ret).to(device),
                                torch.tensor(adv).to(device),
                                torch.tensor(z).to(device))
                        
                        RL_agent.train(sample,process,writer)
                        
                process.write_to_excel()
                RL_agent.save(f"./{hp.file_name}/output/{hp.file_time}/models/") 
                event.set()
                
    finally:
        writer.close()
        


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
                action,_,_,_ = RL_agent.select_action(np.array(state),True)
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


class ReplayBufferManager(BaseManager):
    pass

ReplayBufferManager.register('ReplayBuffer', ReplayBuffer)



class ActorLearner:
    def __init__(self, hp):
        self.hp = hp
        manager = ReplayBufferManager()
        manager.start()
        ctx = mp.get_context('spawn')
        self.replaybuffer = manager.ReplayBuffer(hp.buffer_size, hp.num_processes, hp.num_steps)
        self.counter = ctx.Value('i', -1)  # 共享计数器
        self.event = ctx.Event()  # 事件通知

        # 创建actor和learner进程
        self.actor_process = ctx.Process(target=actor, args=(hp, self.replaybuffer, self.event, self.counter))
        self.learner_process = ctx.Process(target=learner, args=(hp, self.replaybuffer, self.event, self.counter))
        
        
    def start(self):
        # 启动actor和learner进程
        self.actor_process.start()
        self.learner_process.start()

    def join(self):
        # 等待actor和learner进程完成
        self.actor_process.join()
        self.learner_process.join()



