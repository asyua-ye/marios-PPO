import numpy as np
import torch
import multiprocessing as mp
import pickle
import logging
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, action_dim, gae, discount):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.values = torch.zeros(num_steps, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.long)
        self.masks = torch.ones(num_steps , num_processes, 1).to(device)
        self.adv = torch.zeros(num_steps, num_processes, 1).to(device)
        self.gae_tau = gae
        self.discount = discount
        self.num_steps = num_steps
        self.step = 0
        self.rewards_norm = RunningMeanStd(1)
        self.R = torch.zeros(num_processes,1).to(device)
        
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        x = x.view(x.shape[0],-1)
        
        return x

    def insert(self, state, action,action_log_prob, value, reward, mask):
        
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(self.scale(reward,self.rewards_norm,mask))
        self.masks[self.step].copy_(mask)

        
        self.step = (self.step + 1) % self.num_steps
        
    def computeReturn(self,mask):
        """
        每次和环境交互完了用
        ppo不需要next_state
        如果是SAC，直接把下一个状态传过来当作最后一个next_state
        
        """
        
        
        self.returns[-1] = self.rewards[-1] * mask
        for step in reversed(range(len(self.rewards) - 1)):
            self.returns[step] = self.rewards[step] + self.discount * self.returns[step + 1] * self.masks[step]

        # 标准化returns
        returns_mean, returns_std = torch.mean(self.returns), torch.std(self.returns)
        self.returns = (self.returns - returns_mean) / (returns_std + 1e-8)  # 添加小值避免除以零
            
        
        
        
        state=self.states.view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, self.actions.size(-1)).cpu().data.numpy()
        action_log_probs=self.action_log_probs.view(-1,1).cpu().data.numpy()
        returns = self.returns.view(-1,1).cpu().data.numpy()
        
        return (state,action,action_log_probs,returns)


class ReplayBuffer(object):
    def __init__(self, max_size, num_processes, num_steps):
        self.mem = []
        self.memlen = 0
        self.max = max_size
        self.numactor = num_processes
        self.rollout = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.max_size = 0
        self.lock = mp.Lock()

        # 预计算索引
        self.rollout_indices = np.array([i // (self.numactor * self.rollout) for i in range(max_size * self.numactor * self.rollout)])
        self.actor_indices = np.array([i % (self.numactor * self.rollout) for i in range(max_size * self.numactor * self.rollout)])

        # 分开存储各个属性
        self.states = []
        self.actions = []
        self.returns = []

    def push(self, data):
        with self.lock:
            state, action, action_log_probs, returns = data

            if self.memlen < self.max:
                self.states.append(state)
                self.actions.append(action)
                self.returns.append(returns)

            else:
                self.states[self.pos] = state
                self.actions[self.pos] = action
                self.returns[self.pos] = returns

            self.pos = (self.pos + 1) % self.max
            self.memlen = min(self.memlen + 1, self.max)
            
    def getIsFullFlag(self):
        if self.memlen==self.max:
            return True
        else:
            return False
        

    def PPOsample(self, num_mini_batch=40):
        """
        由于 PPO 是 on-policy 的，所以需要对整个回放池进行一次迭代，
        但每次抽取的样本都是随机的且不放回的。
        """
        mini_batch_size = len(self) * self.numactor * self.rollout // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(len(self) * self.numactor * self.rollout)), mini_batch_size, drop_last=False)

        for ind in sampler:
            state = np.array([self.states[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            action = np.array([self.actions[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            action_log_prob = np.array([self.action_log_probs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            adv = np.array([self.advs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            ret = np.array([self.rets[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            
            yield (torch.tensor(state).to(self.device), 
                   torch.tensor(action).to(self.device), 
                   torch.tensor(action_log_prob).to(self.device),
                   torch.tensor(ret).to(self.device),
                   torch.tensor(adv).to(self.device),)    
            
    def sample(self, batch_size, isTrain=True):
        with self.lock:
            total_samples = batch_size
            if isTrain:
                inds = np.random.randint(0, int (self.memlen * self.numactor * self.rollout * 0.8), size=total_samples)
            else:
                inds = np.random.randint(int (self.memlen * self.numactor * self.rollout * 0.8), 
                                         self.memlen * self.numactor * self.rollout, size=total_samples)
                

            # 使用 NumPy 高级索引和列表推导式
            states = np.array([self.states[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            actions = np.array([self.actions[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            returns = np.array([self.returns[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])

            return (
                torch.as_tensor(states),
                torch.as_tensor(actions),
                torch.as_tensor(returns),
            )
            
    def save(self, filename="replay_buffer.pkl"):
        with self.lock:
            data = {
                'states': self.states,
                'actions': self.actions,
                'returns': self.returns,
                'pos': self.pos,
                'memlen': self.memlen,
                'max_size': self.max_size,
            }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filename="replay_buffer.pkl"):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            with self.lock:
                self.states = data['states']
                self.actions = data['actions']
                self.returns = data['returns']
                self.pos = data['pos']
                self.memlen = data['memlen']
                self.max_size = data['max_size']
            logging.info(f"Replay buffer loaded from {filename}")
        except FileNotFoundError:
            logging.warning(f"Replay buffer file not found: {filename}")


    def __len__(self):
        with self.lock:
            return self.memlen  # 返回实际存储的样本数量
