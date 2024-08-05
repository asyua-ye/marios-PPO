import numpy as np
import torch
import h5py
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset

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
        
    def computeReturn(self,next_value,mask):
        """
        每次和环境交互完了用
        ppo不需要next_state
        如果是SAC，直接把下一个状态传过来当作最后一个next_state
        
        """
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta= self.rewards[step] + next_value * self.discount * mask- self.values[step]
            gae = delta+self.discount * self.gae_tau * mask * gae
            self.returns[step] = gae + self.values[step]
            
            next_value = self.values[step]
            if step!=0:
                mask = self.masks[step-1]
            else:
                mask = torch.ones_like(self.masks[step-1])
            
            
        self.adv = self.returns - self.values
        
        self.adv = (self.adv - torch.mean(self.adv)) / (
            torch.std(self.adv) + 1e-5)
        
        state=self.states.view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, self.actions.size(-1)).cpu().data.numpy()
        action_log_probs=self.action_log_probs.view(-1,1).cpu().data.numpy()
        adv = self.adv.view(-1,1).cpu().data.numpy()
        returns = self.returns.view(-1,1).cpu().data.numpy()
        
        return (state,action,action_log_probs,adv,returns)


class ReplayBuffer(object):
    def __init__(self, capacity, discount, train_ratio=0.8, batch_size=32):
        self.capacity = capacity
        self.discount = discount
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.buffer = []
        self.train_loader = None
        self.val_loader = None

    def load_data_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            states = f['states'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            dones = f['dones'][:]

        # 计算returns
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for step in reversed(range(len(rewards) - 1)):
            returns[step] = rewards[step] + self.discount * returns[step + 1] * (1.0 - dones[step])

        # 标准化returns
        returns_mean, returns_std = np.mean(returns), np.std(returns)
        returns = (returns - returns_mean) / (returns_std + 1e-8)  # 添加小值避免除以零

        # 打乱并划分数据
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.train_ratio)

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # 创建训练集和验证集的DataLoader
        self.train_loader = self._create_dataloader(states[train_indices], 
                                                    actions[train_indices], 
                                                    returns[train_indices], 
                                                    shuffle=True)
        
        self.val_loader = self._create_dataloader(states[val_indices], 
                                                  actions[val_indices], 
                                                  returns[val_indices], 
                                                  shuffle=False)

        # 存储所有数据到buffer
        self.buffer = list(zip(states, actions, returns))

    def _create_dataloader(self, states, actions, returns, shuffle):
        dataset = TensorDataset(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(returns, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, 
                          pin_memory=True, num_workers=4)

    def add(self, state, action, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[len(self.buffer) % self.capacity] = (state, action, reward)

    def get_iterator(self, is_train=True):
        return self.train_loader if is_train else self.val_loader

    def __len__(self):
        return len(self.buffer)
