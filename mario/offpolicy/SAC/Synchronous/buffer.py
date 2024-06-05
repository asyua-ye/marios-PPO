import numpy as np
import torch
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, action_dim, discount):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps+1, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.long)
        self.masks = torch.ones(num_steps+1 , num_processes, 1).to(device)
        self.exp = torch.zeros(num_steps, num_processes, 1).to(device)
        
        self.discount = discount
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = device
        self.step = 0
        self.rewards_norm = RunningMeanStd(action_dim)
        self.R = torch.zeros(num_processes,action_dim).to(device)
        
        
        
    def scale(self,x,x_norm,mask):
        
        self.R = mask * self.R
        self.R = self.discount * self.R + x
        x_norm.update(self.R)
        std = torch.tensor(self.rewards_norm.std, dtype=torch.float32, device=x.device)
        x = x / (std + 1e-8)
        
        return x

    def insert(self, state, action, reward, mask):
        
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        self.rewards[self.step].copy_(self.scale(reward,self.rewards_norm,mask))
        self.masks[self.step].copy_(mask)
        self.step = (self.step + 1) % self.num_steps
        
    def lastInsert(self,next_state,next_mask):
        self.states[-1].copy_(next_state)
        self.masks[-1].copy_(next_mask)
        
        
    def computeReturn(self):
        """
        每次和环境交互完了用
        ppo不需要next_state
        如果是SAC，直接把下一个状态传过来当作最后一个next_state
        
        """
        temp=0
        nstep=torch.zeros(self.num_processes,1).to(self.device)
        old=self.rewards
        for i in reversed(range(self.rewards.size(0))):
            temp=temp+self.rewards[i]
            nstep+=1
            self.exp[i]=nstep
            self.rewards[i]=temp
            temp*=self.masks[i]
            l = torch.nonzero(temp.sum(dim=-1)==0).view(-1).tolist()
            if  len(l)!=0:
                nstep[l]=0
                temp[l]+=old[i][l]
                self.rewards[i][l]=temp[l]
                nstep[l]+=1
                self.exp[i][l]=nstep[l]
            temp=temp*self.discount
        
        state=self.states[:-1].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, self.actions.size(-1)).cpu().data.numpy()
        next_state=self.states[1:].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        reward = self.rewards.view(-1, self.actions.size(-1)).cpu().data.numpy()
        mask = self.masks[1:].view(-1,1).cpu().data.numpy()
        exp = self.exp.view(-1,1).cpu().data.numpy()
        
        return (state,action,next_state,reward,mask,exp)


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
        self.masks = []
        self.rewards = []
        self.next_states = []
        self.exps = []

    def push(self, data):
        state, action, next_state, reward, mask, exp = data
        
        if len(self.states) < self.max:
            self.states.append(state)
            self.actions.append(action)
            self.masks.append(mask)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.exps.append(exp)
        else:
            self.states[self.pos] = state
            self.actions[self.pos] = action
            self.masks[self.pos] = mask
            self.rewards[self.pos] = reward
            self.next_states[self.pos] = next_state
            self.exps[self.pos] = exp

        self.pos = (self.pos + 1) % self.max
        

    def sample(self, batch_size, num_epch_train):
        
        total_samples = batch_size * num_epch_train
        inds = np.random.randint(0, len(self) * self.numactor * self.rollout, size=total_samples).reshape(num_epch_train, batch_size)
        
        for ind in inds:
            state = np.array([self.states[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            action = np.array([self.actions[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            mask = np.array([self.masks[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            reward = np.array([self.rewards[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            next_state = np.array([self.next_states[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            exp = np.array([self.exps[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])

            yield (torch.tensor(state).to(self.device), 
                   torch.tensor(action).to(self.device), 
                   torch.tensor(next_state).to(self.device),
                   torch.tensor(reward).to(self.device),
                   torch.tensor(mask).to(self.device),
                   torch.tensor(exp).to(self.device))

    def __len__(self):
        return len(self.states)
