import numpy as np
import torch
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, action_dim, gae, discount):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.values = torch.zeros(num_steps, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.long)
        self.masks = torch.ones(num_steps , num_processes, 1).to(device)
        self.adv = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.gae_tau = gae
        self.discount = discount
        self.num_steps = num_steps
        self.step = 0
        self.rewards_norm = RunningMeanStd(action_dim)
        self.R = torch.zeros(num_processes,action_dim).to(device)
        
        
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
        action_log_probs=self.action_log_probs.view(-1,self.actions.size(-1)).cpu().data.numpy()
        adv = self.adv.view(-1,self.actions.size(-1)).cpu().data.numpy()
        returns = self.returns.view(-1,self.actions.size(-1)).cpu().data.numpy()
        
        return (state,action,action_log_probs,adv,returns)


class ReplayBuffer(object):
    def __init__(self, maxs, num_processes, num_steps):
        self.max = maxs
        self.numactor = num_processes
        self.rollout = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        
        # 分开存储各个属性
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.advs = []
        self.rets = []

    def push(self, data):
        state, action, action_log_prob, adv, ret = data
        
        if len(self.states) < self.max:
            self.states.append(state)
            self.actions.append(action)
            self.action_log_probs.append(action_log_prob)
            self.advs.append(adv)
            self.rets.append(ret)
            
        else:
            self.states[self.pos] = state
            self.actions[self.pos] = action
            self.action_log_probs[self.pos] = action_log_prob
            self.advs[self.pos] = adv
            self.rets[self.pos] = ret
            

        self.pos = (self.pos + 1) % self.max

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


    def __len__(self):
        return len(self.states)
