import numpy as np
import torch
import multiprocessing as mp
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
        self.returns = []
        self.adv = []

    def push(self, data):
        with self.lock:
            state,action,action_log_probs,adv,returns = data

            if self.memlen < self.max:
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(action_log_probs)
                self.returns.append(returns)
                self.adv.append(adv)

            else:
                self.states[self.pos] = state
                self.actions[self.pos] = action
                self.log_probs[self.pos] = action_log_probs
                self.returns[self.pos] = returns
                self.adv[self.pos] = adv

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
            returns = np.array([self.returns[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])
            advs = np.array([self.adv[self.rollout_indices[i]][self.actor_indices[i]] for i in inds])

            return (
                torch.as_tensor(states),
                torch.as_tensor(actions),
                torch.as_tensor(log_probs),
                torch.as_tensor(returns),
                torch.as_tensor(advs),
            )
 
    def __len__(self):
        with self.lock:
            return self.memlen  # 返回实际存储的样本数量
        

def process_batch(batch):
    
    states, actions, log_probs, returns, advs = map(np.array, zip(*batch))
    
    return (
        torch.from_numpy(states),
        torch.from_numpy(actions),
        torch.from_numpy(log_probs),
        torch.from_numpy(returns),
        torch.from_numpy(advs)
    )


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # 使用 np.float64 数据类型
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def add_batch(self, priorities, data):
        assert len(priorities) == len(data), "Priorities and data must have the same length"
        
        for priority, item in zip(priorities, data):
            self.add(priority, item)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def update_batch(self, idxs, priorities):
        idxs = idxs + self.capacity - 1
        changes = priorities - self.tree[idxs]
        self.tree[idxs] = priorities
        
        for idx,change in zip(idxs,changes):
            self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        assert 0 <= dataIdx < self.capacity, f"Invalid dataIdx: {dataIdx}, capacity: {self.capacity}"
        return (dataIdx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.lock = mp.Lock()

        
    def add_batch(self, experiences, priorities=None):
        with self.lock:
            if priorities is None:
                priorities = np.full(len(experiences), self.max_priority ** self.alpha)
            else:
                priorities = np.power(priorities, self.alpha)
            self.tree.add_batch(priorities, experiences)

    def sample(self, batch_size):
        with self.lock:
            batch = []
            idxs = []
            segment = self.tree.total() / batch_size
            priorities = []

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if data == 0:
                    idx = np.random.choice(self.tree.n_entries)
                    pidx = idx + self.tree.capacity - 1
                    p, data = self.tree.tree[pidx], self.tree.data[idx]
                assert 0 <= idx < self.capacity, f"Invalid idx from get: {idx}, capacity: {self.capacity}"
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

            sampling_probabilities = priorities / self.tree.total()
            is_weight = np.power(np.maximum(self.tree.n_entries * sampling_probabilities, 1e-10), -self.beta)
            is_weight /= is_weight.max()

            self.beta = np.min([1., self.beta + self.beta_increment])
            
            processed_batch = process_batch(batch)
            
            return processed_batch, np.array(idxs), is_weight
    
    def update_batch(self, idxs, errors):
        with self.lock:
            # priorities = (errors + self.epsilon) ** self.alpha
            priorities = np.log10(errors + self.epsilon + 1) ** self.alpha  # 对数缩放
            self.tree.update_batch(idxs, priorities)
            self.max_priority = max(self.max_priority, np.max(priorities))


