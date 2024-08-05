import numpy as np
import torch
import multiprocessing as mp
from utils.tool import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer(object):
    def __init__(self, num_steps, num_processes, state_dim, action_dim, gae, discount):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros((num_steps+1, num_processes,) + state_dim).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.values = torch.zeros(num_steps, num_processes, 1).to(device)
        self.returns = torch.zeros(num_steps, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.actions = torch.zeros(num_steps, num_processes, action_dim).to(device, torch.long)
        self.masks = torch.ones(num_steps+1 , num_processes, 1).to(device)
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
        
    def lastInsert(self,next_state,next_mask):
        self.states[-1].copy_(next_state)
        self.masks[-1].copy_(next_mask)
        
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
            
            
        
        state=self.states.view(-1,*self.states.size()[2:]).cpu().data.numpy()
        action=self.actions.view(-1, self.actions.size(-1)).cpu().data.numpy()
        action_log_probs=self.action_log_probs.view(-1,1).cpu().data.numpy()
        returns = self.returns.view(-1,1).cpu().data.numpy()
        rewards = self.rewards.view(-1, 1).cpu().data.numpy()
        next_states = self.states[1:].view(-1,*self.states.size()[2:]).cpu().data.numpy()
        masks = self.masks[1:].view(-1,1).cpu().data.numpy()
        
        
        return (state,action,action_log_probs,returns,rewards,next_states,masks)


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
        self.log_probs = []
        self.rewards = []
        self.returns = []
        self.next_states = []
        self.masks = []

    def push(self, data):
        with self.lock:
            state, action, action_log_probs, returns, rewards, next_states, masks = data

            if self.memlen < self.max:
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(action_log_probs)
                self.returns.append(returns)
                self.rewards.append(rewards)
                self.next_states.append(next_states)
                self.masks.append(masks)
            else:
                self.states[self.pos] = state
                self.actions[self.pos] = action
                self.log_probs[self.pos] = action_log_probs
                self.rewards[self.pos] = rewards
                self.returns[self.pos] = returns
                self.next_states[self.pos] = next_states
                self.masks[self.pos] = masks

            self.pos = (self.pos + 1) % self.max
            self.memlen = min(self.memlen + 1, self.max)

    def sample(self, batch_size, mini_buffer=1):
        with self.lock:
            total_samples = batch_size * mini_buffer
            inds = np.random.randint(0, self.memlen * self.numactor * self.rollout, size=total_samples)

            # 使用 NumPy 高级索引和列表推导式
            states = np.array([self.states[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            actions = np.array([self.actions[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            log_probs = np.array([self.log_probs[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            rewards = np.array([self.rewards[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            returns = np.array([self.returns[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            next_states = np.array([self.next_states[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            masks = np.array([self.masks[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])

            return (
                torch.as_tensor(states),
                torch.as_tensor(actions),
                torch.as_tensor(log_probs),
                torch.as_tensor(returns),
                torch.as_tensor(next_states),
                torch.as_tensor(masks),
                torch.as_tensor(rewards),
            )
 
    def __len__(self):
        with self.lock:
            return self.memlen  # 返回实际存储的样本数量
