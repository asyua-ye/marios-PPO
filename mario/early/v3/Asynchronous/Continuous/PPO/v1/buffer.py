import numpy as np
import torch
import multiprocessing as mp
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
        self.z = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.adv = torch.zeros(num_steps, num_processes, action_dim).to(device)
        self.gae_tau = gae
        self.discount = discount
        self.num_steps = num_steps
        self.step = 0

    def insert(self, state, action,action_log_prob, value, reward, mask,z):
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step].copy_(mask)
        self.z[self.step].copy_(z)

        
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
        z = self.z.view(-1,self.actions.size(-1)).cpu().data.numpy()
        
        return (state,action,action_log_probs,adv,z,returns)


class ReplayBuffer(object):
    def __init__(self,max,num_processes,num_steps):
        self.mem = []
        self.memlen = 0
        self.max = max
        self.numactor = num_processes
        self.rollout = num_steps
        self.device = "cpu"
        self.pos = 0
        self.max_size = 0
        self.lock = mp.Lock()
    
    def push(self,data):
        with self.lock:
            if len(self.mem)<self.max:
                self.mem.append(data)
            else:
                self.mem[int(self.pos)]=(data)
            self.pos = (self.pos + 1) % self.max
            
            self.max_size = len(self.mem) * self.numactor * self.rollout
        
        
    def PPOsample(self, indices, num_mini_batch=40):
        with self.lock:
            max_size = len(indices) * self.numactor * self.rollout
            mini_batch_size = max_size // num_mini_batch
            sampler = BatchSampler(SubsetRandomSampler(range(max_size)), mini_batch_size, drop_last=False)
            mem = [self.mem[i] for i in indices]

            states, actions, log_probs, advantages, zs, returns = zip(*mem)
            samples = []

            for ind in sampler:
                state = np.array([states[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
                action = np.array([actions[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
                action_log_prob = np.array([log_probs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
                adv = np.array([advantages[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
                z = np.array([zs[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])
                ret = np.array([returns[i // (self.numactor * self.rollout)][i % (self.numactor * self.rollout)] for i in ind])

                samples.append((state, action, action_log_prob, ret, adv, z))
            return samples
        
    def sample(self, batch_size):
        with self.lock:
            ind = np.random.randint(0,  self.max_size, size=batch_size)
            state = []
            action = []
            mask = []
            reward= []
            adv = []
            for i in range(batch_size):
                two = ind[i] % (self.numactor*self.rollout)
                one = ind[i] // (self.numactor*self.rollout)
                s, a, m, r, advs=self.mem[one]
                state.append(s[two])
                action.append(a[two])
                mask.append(m[two])
                reward.append(r[two])
                adv.append(advs[two])
                
            return (torch.tensor(np.array(state)).to(self.device), 
                    torch.tensor(np.array(action)).to(self.device), 
                    torch.tensor(np.array(mask)).to(self.device),
                    torch.tensor(np.array(reward)).to(self.device),
                    torch.tensor(np.array(adv)).to(self.device)
                    )
    
    def __len__(self):
        with self.lock:
            return len(self.mem)
