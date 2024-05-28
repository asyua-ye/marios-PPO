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



    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,max_action,share=False,ppg=False,std=False,log_std_min=-20,log_std_max=2):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action
        self.std = std
        self.ppg = ppg
        self.share = share
        
        self.conv1 = nn.Conv2d(self.state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1_1 = nn.Linear(self.feature_size()+action_dim, 512)
        self.q1 = nn.Linear(512, action_dim)
        
        self.fc1_2 = nn.Linear(self.feature_size()+action_dim, 512)
        self.q2 = nn.Linear(512, action_dim)
        
        
        
        self.fc2 = nn.Linear(self.feature_size(), 512)
        self.mean_linear = nn.Linear(512, action_dim)
        
        if self.std:
            #这样所有批量都用一个std吗？
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.fc3 = nn.Linear(self.feature_size(), 512)
            self.log_std_linear = nn.Linear(512, action_dim)  
        
        self.state_norm = RunningMeanStd(self.feature_size())
        
        self._initialize_weights()


    def forward(self, x, action=None):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.norm(x, self.state_norm)
            
            
        if action is not None:    
        
            sa = torch.cat([x, action], 1)
            q1 = F.tanh(self.fc1_1(sa))
            q1 = self.q1(q1)
            
            
            q2 = F.tanh(self.fc1_2(sa))
            q2 = self.q2(q2)
            
            return q1,q2
        
        else:
            x_actor = x
            if self.ppg:
                x_actor = x.detach()
            
            a = F.tanh(self.fc2(x_actor))
            mean    = self.mean_linear(a)
            
            if not self.std:
                if self.share:
                    log_std = self.log_std_linear(a).clamp(self.log_std_min, self.log_std_max)
                else:    
                    s = F.tanh(self.fc3(x_actor))
                    log_std = self.log_std_linear(s).clamp(self.log_std_min, self.log_std_max)
            else:
                log_std = self.log_std * torch.ones(*mean.shape)
                    
            return mean, log_std
    
    def getAction(self,state,deterministic=False,with_logprob=True,rsample=True):
        
        mean, log_std= self.forward(state)
        
        
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if deterministic:
            z = mean
        else:
            if rsample:
                z = normal.rsample()
            else:
                z = normal.sample()
                
                
                
        action = torch.sigmoid(z)
        
        if with_logprob:
            log_prob = normal.log_prob(z)
            # log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z)))
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        else:
            log_prob = None
            
        action = self.max_action*action
        
        
        return action,log_prob
    
    def getQ(self,state,action):
        q1,q2= self.forward(state,action)
        
        return q1,q2
    
    def getLogprob(self,state,action):
        
        mean, log_std= self.forward(state)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        action = action / self.max_action
        
        old_z = torch.atanh(torch.clamp(action, min=-0.999999, max=0.999999))
        
        old_logprob = normal.log_prob(old_z) - torch.log(1 - action.pow(2) + 1e-6)
    
        return old_logprob
        

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.state_dim)))).view(1, -1).size(1)
    
    def norm(self, x, x_norm):
        x_norm.update(x.detach())
        x = x_norm.normalize(x)
        return x

    def _initialize_weights(self):
        
        if self.std:
            nn.init.constant_(self.log_std, 0)
        
        for name, module in self.named_modules():
            if hasattr(module, 'weight'):
                if name == 'conv1' or name == 'conv1' or name == 'conv1':
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                elif name == 'mean_linear' or (name == 'log_std_linear' and not self.std):
                    nn.init.orthogonal_(module.weight, 0.01)
                elif name == 'q1' or name == 'q2':
                    nn.init.orthogonal_(module.weight, 1.0)
                else:
                    nn.init.orthogonal_(module.weight, nn.init.calculate_gain('tanh'))
                    # nn.init.xavier_uniform_(module.weight)
                    # nn.init.kaiming_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    

class agent(object):
    def __init__(self,state_dim, action_dim,max_action,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actorCritic = ActorCritic(self.state_dim,self.action_dim,max_action,hp.share,hp.ppg,hp.std,hp.log_std_min,hp.log_std_max).to(self.device)
        self.Q_target = copy.deepcopy(self.actorCritic)
        self.actorCritic_o = torch.optim.Adam(self.actorCritic.parameters(),lr=hp.actor_lr,eps=hp.eps)
        self.replaybuffer = buffer.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        
        lambda_lr = lambda step: 1 - step / hp.max_steps if step < hp.max_steps else 0
        self.scheduler = LambdaLR(self.actorCritic_o, lr_lambda=lambda_lr)
        self.dicount = hp.discount
        
        #SAC
        self.batch = hp.batch
        self.num_epch_train = hp.num_epch_train
        self.adaptive_alpha = hp.adaptive_alpha
        self.tau = hp.tau
        self.grad = hp.grad
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
        action,logprob = self.actorCritic.getAction(state,deterministic)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        logprob = logprob.view(-1,self.action_dim).cpu().data.numpy()
        
        return action
    
    
    
    
    def train(self,process,writer):
        
        for i in range(self.num_epch_train):
            sample = self.replaybuffer.sample(self.batch)
            
            
            self.learn_step += 1
            state, action,next_state,reward,mask,exp=sample
            
            
            
            ####################
            #updata  actor
            ####################
            new_action, log_prob= self.actorCritic.getAction(state)
            new_q1,new_q2 = self.actorCritic.getQ(state,new_action)
            new_q = torch.min(new_q1,new_q2)
            actor_loss = (self.alpha*log_prob - new_q).mean()
            
            
            
            self.actorCritic_o.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), self.grad)
            self.actorCritic_o.step()
            
            ####################
            #updata  Q
            ####################
            q1,q2 = self.actorCritic.getQ(state,action)
            
            with torch.no_grad():
                next_action, log_next_prob= self.actorCritic.getAction(next_state)
                target_q1,target_q2 = self.Q_target.getQ(next_state,next_action)
                target_q = torch.min(target_q1,target_q2)
                target_value = target_q - self.alpha * log_next_prob
                next_q_value = reward + mask * (self.dicount**exp) * target_value
                
            
            
            q1_loss = ((q1 - next_q_value)**2).mean()
            q2_loss = ((q2 - next_q_value)**2).mean()
            q_loss = q1_loss + q2_loss
            
            self.actorCritic_o.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), self.grad)
            self.actorCritic_o.step()
            
            self.scheduler.step()
            
            
            ####################
            #soft updata  valuetarget
            ####################
            with torch.no_grad():
                for target_param,param in zip(self.Q_target.parameters(),self.actorCritic.parameters()):
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
                
    
    
    def save(self,filename):
        torch.save(self.actorCritic.state_dict(),filename+"_actorCritic")
        torch.save(self.actorCritic_o.state_dict(),filename+"_actorCritic_optim")
        
        
        
    def load(self,filename):
        self.actorCritic.load_state_dict(torch.load(filename+"_actorCritic"))
        # self.actorCritic_o.load_state_dict(torch.load(filename+"_actorCritic_optim"))
        
    def IsCheckpoint(self,Score):
        if self.Maxscore<Score:
            self.Maxscore = Score
            return True
        else:
            return False
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
