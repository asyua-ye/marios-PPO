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
    def __init__(self,max,num_processes,num_steps):
        self.mem = []
        self.memlen = 0
        self.max = max
        self.numactor = num_processes
        self.rollout = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = 0
        self.max_size = 0
    
    def push(self,data):
        if len(self.mem)<self.max:
            self.mem.append(data)
        else:
            self.mem[int(self.pos)]=(data)
        self.pos = (self.pos + 1) % self.max
        
        self.max_size = len(self.mem) * self.numactor * self.rollout
        
        
    def PPOsample(self, num_mini_batch=40):
        """
        由于 PPO 是 on-policy 的，所以需要对整个回放池进行一次迭代，
        但每次抽取的样本都是随机的且不放回的。
        """
        mini_batch_size = self.max_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(self.max_size)), mini_batch_size, drop_last=False)

        # 提前从内存中提取所有数据
        states, actions, log_probs, advantages, zs, returns = zip(*self.mem)

        for ind in sampler:
            state = np.array([states[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            action = np.array([actions[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            action_log_prob = np.array([log_probs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            adv = np.array([advantages[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            z = np.array([zs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            ret = np.array([returns[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
            
            
            yield (torch.tensor(state).to(self.device), 
                   torch.tensor(action).to(self.device), 
                   torch.tensor(action_log_prob).to(self.device),
                   torch.tensor(ret).to(self.device),
                   torch.tensor(adv).to(self.device),
                   torch.tensor(z).to(self.device),
                   )
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.max_size, size=batch_size)
        
        state = np.empty((batch_size, *self.mem[0][0][0].shape), dtype=np.float32)
        action = np.empty((batch_size, *self.mem[0][1][0].shape), dtype=np.float32)
        mask = np.empty((batch_size, *self.mem[0][4][0].shape), dtype=np.float32)
        reward = np.empty((batch_size,*self.mem[0][3][0].shape), dtype=np.float32)
        next_state = np.empty((batch_size, *self.mem[0][2][0].shape), dtype=np.float32)
        exp = np.empty((batch_size, *self.mem[0][5][0].shape), dtype=np.float32)

        for i in range(batch_size):
            two = ind[i] % (self.numactor * self.rollout)
            one = ind[i] // (self.numactor * self.rollout)
            states, actions, next_states, rewards, masks, exps = self.mem[one]
            state[i] = states[two]
            action[i] = actions[two]
            mask[i] = masks[two]
            reward[i] = rewards[two]
            next_state[i] = next_states[two]
            exp[i] = exps[two]

        return (torch.tensor(state).to(self.device), 
                torch.tensor(action).to(self.device), 
                torch.tensor(next_state).to(self.device),
                torch.tensor(reward).to(self.device),
                torch.tensor(mask).to(self.device),
                torch.tensor(exp).to(self.device))
    
    def __len__(self):
        return len(self.mem)
