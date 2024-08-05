import numpy as np
import torch
import gymnasium as gym
import gym_super_mario_bros
import os
import time
import datetime
import res18
import multiprocessing as mp
from utils.tool import log_and_print,DataProcessor
from buffer import RolloutBuffer,ReplayBuffer
from pre_env import ProcessEnv
from dataclasses import dataclass, asdict
from subproc_vec_env import SubprocVecEnv
from pre_env import ProcessEnv
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.managers import BaseManager
from load_model import getTeacher


def actor(hp, replaybuffer, counter):
    eval_env = gym_super_mario_bros.make(hp.env)
    eval_env = ProcessEnv(eval_env)
    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n
    
    rollout = RolloutBuffer(hp.num_steps, hp.num_processes, state_dim, action_dim, hp.gae, hp.discount)
    
    rds = 0
    episode_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    final_rewards = np.zeros(hp.num_processes, dtype=np.float64)
    rounds = np.zeros(hp.num_processes, dtype=np.float64)
    start_time = time.time()
    train = []
    log_and_print(train, (f"begin time:  {hp.file_time}\n"))
    if hp.checkpoint and not os.path.exists(f"./output/{hp.file_name}/{hp.file_time}/checkpoint/models"):
        os.makedirs(f"./output/{hp.file_name}/{hp.file_time}/checkpoint/models")

    output_directory = f"./output/{hp.file_name}/{hp.file_time}/models/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        hp.save_to_file(os.path.join(output_directory, 'hyperparameters.txt'))
    process = DataProcessor(f'./output/{hp.file_name}/{hp.file_time}/actor_')

    t = 0
    # teachers
    teachers = getTeacher(hp.teachers,hp)
    RL_agent = teachers[0]
    s = hp.teachers[0]
    world = s // 4 + 1
    stage = s % 4 + 1
    env_name = f"SuperMarioBros-{world}-{stage}-v0"
    env = SubprocVecEnv(hp.num_processes, env_name)
    state = env.reset()
    count = counter.value
    while True:
        if t >= 0:

            if counter.value == -2:
                env.close()
                file_time1 = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                log_and_print(train, f"\nend time: {file_time1}")
                np.savetxt(f"./output/{hp.file_name}/{hp.file_time}/train.txt", train, fmt='%s')
                break
            
            
            if count!=counter.value:
                env.close()
                count = counter.value
                RL_agent = teachers[count]
                s = hp.teachers[count]
                world = s // 4 + 1
                stage = s % 4 + 1
                env_name = f"SuperMarioBros-{world}-{stage}-v0"
                env = SubprocVecEnv(hp.num_processes, env_name)
                state = env.reset()
                replaybuffer.memlen = 0
                replaybuffer.pos = 0
                
            
            total_reward = []
            fin = 0   
                
            t += 1
            s1 = time.time()
            for step in range(hp.num_steps):
                rounds += 1
                action, logit, value = RL_agent.select_action(np.array(state))
                next_state, reward, ep_finished, info = env.step(action)
                episode_rewards += reward
                mask = 1. - ep_finished.astype(np.float32)
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

                rollout.insert(state, action, logit, value, reward, mask)
                state = next_state
                if torch.any(mask == 0).item() and np.any(final_rewards != 0):
                    for i, fo in enumerate(info):
                        if mask[i] == 0:
                            if fo['flag_get']:
                                fin += 1

            e = time.time()
            if hp.checkpoint:
                rs = len(total_reward)
                if fin > rs:
                    fin = rs
                if rs!=0:
                    fin = fin/ rs *100
                    score = np.mean(total_reward) + np.min(total_reward) + np.max(total_reward) + np.median(total_reward) - np.std(total_reward)
                else:
                    fin = 0
                    score = 0.0
                flag = RL_agent.IsCheckpoint((fin,score))
                log_and_print(train, "---------------------------------------")
                log_and_print(train, f"total {rs}  This Score：{(fin,score)} Max Score:{RL_agent.Maxscore}")
                
            sample_time = (e-s1)
            log_and_print(train, f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
            log_and_print(train, "---------------------------------------")
            s = time.time()
            states, actions, action_log_probs,returns = rollout.computeReturn(mask)
            data = (np.copy(states), np.copy(actions), np.copy(action_log_probs), np.copy(returns))
            replaybuffer.push(data)
            e = time.time()
            # 每次循环结束时保存训练日志
            np.savetxt(f"./output/{hp.file_name}/{hp.file_time}/sample.txt", train, fmt='%s')
            process.process_input(sample_time,'sample_time(s)','time/')
            e = time.time()
            total_time = (e-s1)
            process.process_input(total_time,'total_time(s)','time/')
            process.process_input(t,'Epoch')
            process.write_to_excel()

            
            if counter.value ==-1:
                with counter.get_lock():
                    counter.value = 0
                    
                    
def learner(hp, replaybuffer, counter):
    writer = SummaryWriter(f'./output/{hp.file_name}/{hp.file_time}/{hp.file_time}-res18-{hp.total_timesteps}')
    process = DataProcessor(f'./output/{hp.file_name}/{hp.file_time}/learner_')
    
    s = hp.teachers[0]
    world = s // 4 + 1
    stage = s % 4 + 1
    env_name = f"SuperMarioBros-{world}-{stage}-v0"
    eval_env = ProcessEnv(gym_super_mario_bros.make(env_name,render_mode='human', apply_api_compatibility=True))
    
    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n
    RL_agent = res18.agent(state_dim, action_dim, hp)
    eval_env = SubprocVecEnv(hp.num_processes, env_name)
    t = 1
    batch_size = hp.batch_size
    start_time = time.time()
    file_time = hp.file_time
    texts = []
    
    try:
        while True:
            with counter.get_lock():
                count = counter.value
            if t % hp.total_timesteps == 0 or count==len(hp.teachers):
                counter.value = -2
                break
            
            if count != -1:
                t += 1
                s = time.time()
                total_acc = []
                for _ in range(hp.num_epch_train):
                    sample = replaybuffer.sample(batch_size)
                    acc = RL_agent.Imitate(sample,writer)
                    total_acc.append(acc)
                log_and_print(texts, "---------------------------------------")
                log_and_print(texts,f"Epoch {t},  Average acc: {np.mean(total_acc)*100:.2f}%")
                log_and_print(texts, "---------------------------------------")
                    
                e = time.time()
                train_time = (e - s)
                # eval
                s = time.time()
                text,total_reward = maybe_evaluate_and_print(RL_agent, eval_env, t, start_time,file_time, hp)
                texts.append(text)
                if RL_agent.Maxscore[0]>70:
                    with counter.get_lock():
                        counter.value += 1
                    RL_agent.Maxscore = (0.0,0)
                    count = counter.value
                    s = hp.teachers[count]
                    world = s // 4 + 1
                    stage = s % 4 + 1
                    env_name = f"SuperMarioBros-{world}-{stage}-v0"
                    eval_env = SubprocVecEnv(hp.num_processes, env_name)
                    
                e = time.time()
                eval_time = (e - s)
                process.process_input(total_acc,'acc','train/')
                process.process_input(total_reward,'return','eval/')
                process.process_input(train_time,'train_time','time/')
                process.process_input(eval_time,'eval_time','time/')
                process.process_input(count,'num','stage/')
                process.write_to_excel()
                RL_agent.save(f"./output/{hp.file_name}/{hp.file_time}/models/") 
                texts = [str(text) for text in texts]
                np.savetxt(f"./output/{hp.file_name}/{hp.file_time}/train.txt", texts, fmt='%s')
                
    finally:
        writer.close()
        
        
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
    
    
class ReplayBufferManager(BaseManager):
    pass

ReplayBufferManager.register('ReplayBuffer', ReplayBuffer)



class ActorLearner:
    def __init__(self, hp):
        self.hp = hp
        manager = ReplayBufferManager()
        manager.start()
        ctx = mp.get_context('spawn')
        self.replaybuffer = manager.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        self.counter = ctx.Value('i', -1)  # 共享计数器

        # 创建actor和learner进程
        self.actor_process = ctx.Process(target=actor, args=(hp, self.replaybuffer, self.counter))
        self.learner_process = ctx.Process(target=learner, args=(hp, self.replaybuffer, self.counter))
        
        
    def start(self):
        # 启动actor和learner进程
        self.actor_process.start()
        self.learner_process.start()

    def join(self):
        # 等待actor和learner进程完成
        self.actor_process.join()
        self.learner_process.join()


