import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import buffer
import numpy as np
import math
from utils.tool import RunningMeanStd
from torch.distributions import Normal
from torch.optim.lr_scheduler import LambdaLR



class CNN(nn.Module):
    def __init__(self, state_dim):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        
        self.feature_size = self._get_feature_size(state_dim)
        self.state_norm = RunningMeanStd(self.feature_size)
        
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.norm(x, self.state_norm)
        
        return x
    
    def norm(self, x, x_norm):
        x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x
    
    def _get_feature_size(self, state_dim):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *state_dim)))).view(1, -1).size(1)
    
    def _initialize_weights(self):
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                

class Actor(nn.Module):
    def __init__(self, cnn, action_dim,ppg=False,std=False,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()
        
        self.cnn = cnn
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.std = std
        self.ppg = ppg
        self.fc1 = nn.Linear(self.cnn.feature_size, 512)
        self.fc1_1 = nn.Linear(512, 512)
        self.mean_linear = nn.Linear(512, action_dim)
        
        if self.std:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.log_std_linear = nn.Linear(512, action_dim)
            
        self._initialize_weights()
        
    def forward(self, x):
        x = self.cnn(x)
        if self.ppg:
            x = x.detach()
        x_actor = F.tanh(self.fc1(x))
        x_actor = F.tanh(self.fc1_1(x_actor))
        mean = self.mean_linear(x_actor)
        
        return mean
    
    def get_action(self, state, deterministic=False, rsample=True):
        logits = self.forward(state) 
     
        probabilities = torch.sigmoid(logits)
        
        # action = (probabilities >= self.threshold).int()
        action = torch.bernoulli(probabilities).int()
        
        log_prob = torch.log(probabilities * action + (1 - probabilities) * (1 - action))
        log_prob = log_prob.sum(-1,keepdim=True)
        
        return action, log_prob
    
    def _initialize_weights(self):
        
        if self.std:
            nn.init.constant_(self.log_std, 0)
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'mean_linear' or (name == 'log_std_linear' and not self.std):
                    nn.init.orthogonal_(module.weight, 0.01)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('tanh'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)



class Critic(nn.Module):
    def __init__(self, cnn, action_dim):
        super(Critic, self).__init__()
        
        self.cnn = cnn
        self.fc1_1 = nn.Linear(self.cnn.feature_size + action_dim, 512)
        self.fc1_1_1 = nn.Linear(512, 512)
        self.q1 = nn.Linear(512, 1)
        
        self.fc1_2 = nn.Linear(self.cnn.feature_size + action_dim, 512)
        self.fc1_2_1 = nn.Linear(512, 512)
        self.q2 = nn.Linear(512, 1)
        
        self._initialize_weights()
        
    def forward(self, x, action):
        x = self.cnn(x)
        sa = torch.cat([x, action], 1)
        q1 = F.tanh(self.fc1_1(sa))
        q1 = F.tanh(self.fc1_1_1(q1))
        q1 = self.q1(q1)
        
        q2 = F.tanh(self.fc1_2(sa))
        q2 = F.tanh(self.fc1_2_1(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def get_q(self, state, action):
        q1, q2 = self.forward(state, action)
        return q1, q2
    
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'q1' or name == 'q2':
                    nn.init.orthogonal_(module.weight, 1.0)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('tanh'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)



class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        cnn_net = CNN(state_dim)
        self.actor = Actor(cnn_net,self.action_dim,hp.ppg,hp.std,hp.log_std_min,hp.log_std_max).to(self.device)
        self.critic = Critic(cnn_net,action_dim).to(self.device)
        self.Q_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr,eps=hp.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr,eps=hp.eps)
        
        
        self.replaybuffer = buffer.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        
        lambda_lr = lambda step: 1 - step / hp.max_steps if step < hp.max_steps else 0
        self.scheduler = [LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr),
                          LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)]
        self.dicount = hp.discount
        
        #SAC
        self.batch = hp.batch
        self.num_epch_train = hp.num_epch_train
        self.adaptive_alpha = hp.adaptive_alpha
        self.tau = hp.tau
        self.grad = hp.grad
        self.mp = hp.MP
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=hp.actor_lr)
        else:
            self.alpha = hp.alpha
        
        
        #checkpoint
        self.Maxscore = 0
        self.learn_step = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        if state.ndim == 3:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        action,logprob = self.actor.get_action(state,deterministic)
        
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        logprob = logprob.view(-1,1).cpu().data.numpy()
        
        return action
    
    
    
    def train(self,process,writer):
        
        for i, sample in enumerate(self.replaybuffer.sample(self.batch, self.num_epch_train), 
                                   start=0):
            
            self.learn_step += 1
            state, action,next_state,reward,mask,exp = sample
            
            
            
            ####################
            #updata  actor
            ####################
            new_action, log_prob= self.actor.get_action(state)
            
            if self.mp:
                new_action_flat = new_action.flatten()
                new_action_one_hot = torch.zeros(state.shape[0] * self.action_dim, self.action_dim).to(new_action.device)
                indices = torch.arange(state.shape[0] * self.action_dim)
                column_indices = indices % self.action_dim
                new_action_one_hot[indices, column_indices] = new_action_flat

                state_for_one_hot = state.repeat(*([self.action_dim] + [1] * (state.dim() - 1)))
                new_q1, new_q2 = self.critic.get_q(state_for_one_hot, new_action_one_hot)
                
                assert new_q1.shape == (state.shape[0] * self.action_dim, self.action_dim)
                assert new_q2.shape == (state.shape[0] * self.action_dim, self.action_dim)

                indices = (indices % 7).view(-1,1).to(self.device)
                new_q1 = new_q1.gather(1, indices).view(state.shape[0], self.action_dim)
                new_q2 = new_q2.gather(1, indices).view(state.shape[0], self.action_dim)

            else:
                new_q1, new_q2 = self.critic.get_q(state, new_action)
            
            new_q = torch.min(new_q1,new_q2)
            actor_loss = (self.alpha*log_prob - new_q).mean()
            
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
            self.actor_optimizer.step()
            
            ####################
            #updata  Q
            ####################
            q1,q2 = self.critic.get_q(state,action)
            
            with torch.no_grad():
                next_action, log_next_prob= self.actor.get_action(next_state)
                target_q1,target_q2 = self.Q_target.get_q(next_state,next_action)
                target_q = torch.min(target_q1,target_q2)
                target_value = target_q - self.alpha * log_next_prob
                next_q_value = reward + mask * (self.dicount**exp) * target_value
                
            
            
            q1_loss = ((q1 - next_q_value)**2).mean()
            q2_loss = ((q2 - next_q_value)**2).mean()
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
            self.critic_optimizer.step()
            
            for scheduler in self.scheduler:
                scheduler.step()
            
            
            ####################
            #soft updata  valuetarget
            ####################
            with torch.no_grad():
                for target_param,param in zip(self.Q_target.parameters(),self.critic.parameters()):
                    target_param.data.copy_(
                    target_param.data *(1 - self.tau)  + param.data * self.tau
                )
                
                
            ####################
            #alpha
            ####################
            if self.adaptive_alpha:
                # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()
            
            
            writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
            writer.add_scalar('value_loss', q_loss.item(), global_step=self.learn_step)
            
            if i % self.num_epch_train == 0:
                process.process_input(self.learn_step, 'learn_step', 'train/')
                process.process_input(actor_loss.item(), 'actor_loss', 'train/')
                process.process_input(q_loss.item(), 'q_loss', 'train/')
                process.process_input(new_q.detach().cpu().numpy(), 'new_q', 'train/')
                process.process_input(target_q.detach().cpu().numpy(), 'target_q', 'train/')
                process.process_input(new_action.detach().cpu().numpy(), 'new_action', 'train/')
                process.process_input(log_prob.detach().cpu().numpy(), 'log_prob', 'train/')
                
    
    
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'running_mean_std_state': {
                'n': self.actor.cnn.state_norm.n,
                'mean': self.actor.cnn.state_norm.mean,
                'S': self.actor.cnn.state_norm.S,
                'std': self.actor.cnn.state_norm.std
            }
        }, filename + "_actor")
        
        # 如果需要保存优化器状态，可以取消注释以下代码
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        # torch.save({
        #     'critic_state_dict': self.critic.state_dict(),
        #     'running_mean_std_state': {
        #         'n': self.critic.cnn.state_norm.n,
        #         'mean': self.critic.cnn.state_norm.mean,
        #         'S': self.critic.cnn.state_norm.S,
        #         'std': self.critic.cnn.state_norm.std
        #     }
        # }, filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        
        
    def load(self, filename):
        checkpoint = torch.load(filename + "_actor")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        # 恢复 RunningMeanStd 的状态
        self.actor.cnn.state_norm.n = checkpoint['running_mean_std_state']['n']
        self.actor.cnn.state_norm.mean = checkpoint['running_mean_std_state']['mean']
        self.actor.cnn.state_norm.S = checkpoint['running_mean_std_state']['S']
        self.actor.cnn.state_norm.std = checkpoint['running_mean_std_state']['std']
        
        # 如果需要加载优化器状态，可以取消注释以下代码
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.critic.cnn.state_norm.n = checkpoint['running_mean_std_state']['n']
        # self.critic.cnn.state_norm.mean = checkpoint['running_mean_std_state']['mean']
        # self.critic.cnn.state_norm.S = checkpoint['running_mean_std_state']['S']
        # self.critic.cnn.state_norm.std = checkpoint['running_mean_std_state']['std']
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        
    def IsCheckpoint(self,Score):
        if self.Maxscore<Score:
            self.Maxscore = Score
            return True
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
