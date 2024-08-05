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
    def __init__(self, cnn, action_dim,ppg=False,log_std_min=-20,log_std_max=2):
        super(Actor, self).__init__()
        
        self.cnn = cnn
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.ppg = ppg
        self.fc1 = nn.Linear(self.cnn.feature_size, 512)
        self.fc1_1 = nn.Linear(512, 512)
        self.mean_linear = nn.Linear(512, action_dim)
            
        self._initialize_weights()
        
    def forward(self, x):
        x = self.cnn(x)
        if self.ppg:
            x = x.detach()
        x_actor = F.relu(self.fc1(x))
        x_actor = F.relu(self.fc1_1(x_actor))
        mean = self.mean_linear(x_actor)
        
        
        return mean
    
    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = F.log_softmax(logits, dim=1)
        action = dist.sample().view(-1, 1)
        action_log_probs = log_probs.gather(1, action.long())
        
        return action,action_log_probs
    
    def getLogprob(self,state,action):
        
        logits = self.forward(state)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = F.log_softmax(logits, dim=1)
        old_logprob = log_probs.gather(1, action.long())
        distentroy = dist.entropy().mean()
        
        return old_logprob,distentroy
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'mean_linear' :
                    nn.init.orthogonal_(module.weight, 0.01)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        



class Critic(nn.Module):
    def __init__(self, cnn):
        super(Critic, self).__init__()
        
        self.cnn = cnn
        self.fc1_1 = nn.Linear(self.cnn.feature_size, 512)
        self.fc1_1_1 = nn.Linear(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.cnn(x)
        v = F.tanh(self.fc1_1(x))
        v = F.tanh(self.fc1_1_1(v))
        v = self.critic_linear(v)
        
        
        return v
    
    def getValue(self,state):
        value= self.forward(state)
        
        return value
    
    
    def _initialize_weights(self):
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'critic_linear' :
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
        self.actor = Actor(cnn_net,self.action_dim,hp.ppg,hp.log_std_min,hp.log_std_max).to(self.device)
        self.critic = Critic(cnn_net).to(self.device)
        self.Q_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr,eps=hp.eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr,eps=hp.eps)
        self.replaybuffer = buffer.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        
        lambda_lr = lambda step: 1 - step / hp.max_steps if step < hp.max_steps else 0
        self.scheduler = [LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr),
                          LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)]
        
        #PPO
        self.ppo = hp.ppo_update
        self.clip = hp.clip
        self.grad = hp.grad
        self.num_mini_batch = hp.mini_batch
        self.entropy = hp.entropy
        self.value_weight = hp.value
        self.actor_weight = hp.actor
        
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
        value = self.critic.getValue(state)
        action = action.view(-1,1).cpu().data.numpy()
        logprob = logprob.view(-1,1).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        
        return action,logprob,value
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        value = self.critic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    
    def evaluate_actions(self, state,actions):
        
        logprob,dist_entropy = self.actor.getLogprob(state,actions)
        
        
        return logprob, dist_entropy
    
    
    def train(self,process,writer):
        
        for i in range(self.ppo):
            data_generator = self.replaybuffer.PPOsample(self.num_mini_batch)
            
            for sample in data_generator:
                self.learn_step += 1
                state,action,old_action_log_probs,returns,advs = sample
                
                action_log_probs, dist_entropy = self.evaluate_actions(state,action)
                
                ratio =  torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
                actor_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()
                
                
                actor_loss = self.actor_weight * actor_loss - self.entropy * dist_entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad)
                self.actor_optimizer.step()
                
                values = self.critic.getValue(state)
                
                value_loss = F.mse_loss(returns.mean(-1,keepdim=True), values)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad)
                self.critic_optimizer.step()
                
                for scheduler in self.scheduler:
                    scheduler.step()
                
                
                writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
                writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
                
            if i == self.ppo - 1:
                process.process_input(self.learn_step, 'learn_step', 'train/')
                process.process_input(actor_loss.item(), 'actor_loss', 'train/')
                process.process_input(value_loss.item(), 'value_loss', 'train/')
                process.process_input(dist_entropy.item(), 'dist_entropy', 'train/')
                process.process_input(action_log_probs.detach().cpu().numpy(), 'action_log_probs', 'train/')
                process.process_input(old_action_log_probs.detach().cpu().numpy(), 'old_action_log_probs', 'train/')
                process.process_input(ratio.detach().cpu().numpy(), 'ratio', 'train/')
                process.process_input(surr1.detach().cpu().numpy(), 'surr1', 'train/')
                process.process_input(values.detach().cpu().numpy(), 'values', 'train/')
                process.process_input(returns.detach().cpu().numpy(), 'returns', 'train/')
                
    
    
    def save(self,filename):
        torch.save(self.actor.state_dict(),filename+"_actor")
        torch.save(self.actor_optimizer.state_dict(),filename+"_actor_optimizer")
        torch.save(self.critic.state_dict(),filename+"_critic")
        torch.save(self.critic_optimizer.state_dict(),filename+"_critic_optimizer")
        
        
        
    def load(self,filename):
        self.actor.load_state_dict(torch.load(filename+"_actor"))
        # self.actor_optimizer.load_state_dict(torch.load(filename+"_actor_optimizer"))
        
        
    def IsCheckpoint(self,Score):
        if self.Maxscore<Score:
            self.Maxscore = Score
            return True
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
