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
        self.fc1_1_1 = nn.Linear(512, 512)
        self.q1 = nn.Linear(512, action_dim)
        
        self.fc1_2 = nn.Linear(self.feature_size()+action_dim, 512)
        self.fc1_2_1 = nn.Linear(512, 512)
        self.q2 = nn.Linear(512, action_dim)
        
        
        
        self.fc2 = nn.Linear(self.feature_size(), 512)
        self.fc2_1_1 = nn.Linear(512, 512)
        self.mean_linear = nn.Linear(512, action_dim)
        
        
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
            q1 = F.tanh(self.fc1_1_1(q1))
            q1 = self.q1(q1)
            
            
            q2 = F.tanh(self.fc1_2(sa))
            q2 = F.tanh(self.fc1_2_1(q2))
            q2 = self.q2(q2)
            
            return q1,q2
        
        else:
            x_actor = x
            if self.ppg:
                x_actor = x.detach()
            
            a = F.tanh(self.fc2(x_actor))
            a = F.tanh(self.fc2_1_1(a))
            mean    = self.mean_linear(a)
            
                    
            return mean
    
    def getAction(self,state):
        
        mean= self.forward(state)  
        action = torch.sigmoid(mean)
        # action = torch.tanh(z)
        action = self.max_action*action
        
        return action
    
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
        self.target = copy.deepcopy(self.actorCritic)
        self.actorCritic_o = torch.optim.Adam(self.actorCritic.parameters(),lr=hp.actor_lr,eps=hp.eps)
        self.replaybuffer = buffer.ReplayBuffer(hp.buffer_size,hp.num_processes,hp.num_steps)
        
        lambda_lr = lambda step: 1 - step / hp.max_steps if step < hp.max_steps else 0
        self.scheduler = LambdaLR(self.actorCritic_o, lr_lambda=lambda_lr)
        self.dicount = hp.discount
        self.max_action = max_action
        #TD3
        self.batch = hp.batch
        self.num_epch_train = hp.num_epch_train
        self.tau = hp.tau
        self.grad = hp.grad
        self.mp = hp.MP
        self.noiseClip = hp.noiseclip
        self.updaeActor = hp.update_actor
        self.actionNoise = hp.actionNoise
        self.exNoise = hp.exNoise
        
        #LAP
        self.min_priority = hp.min_priority
        self.LAPalpha = hp.LAPalpha
        
        
        
        #checkpoint
        self.Maxscore = 0
        self.learn_step = 0
        
        
        
    @torch.no_grad()
    def select_action(self,state,deterministic=False):
        
        if state.ndim == 3:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        action = self.actorCritic.getAction(state)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        if not deterministic:
            exnoise = np.random.uniform(0, self.exNoise, size=action.shape[0])
            temp = np.random.normal(0, self.max_action * exnoise[:, np.newaxis], size=(action.shape[0], self.action_dim))
            action = (action+ temp).clip(0, self.max_action)
        
        return action
    
    
    
    
    def train(self,process,writer):
        
        for i in range(self.num_epch_train):
            sample = self.replaybuffer.sample(self.batch)
            
            
            self.learn_step += 1
            state, action,next_state,reward,mask,exp=sample
            
            actor_loss = torch.zeros(1)
            new_action = torch.zeros_like(action)
            new_q = torch.zeros_like(action)
            if self.learn_step % self.updaeActor == 0:
                ####################
                #updata  actor
                ####################
                new_action= self.actorCritic.getAction(state)
                
                if self.mp:
                    new_action_flat = new_action.flatten()
                    new_action_one_hot = torch.zeros(state.shape[0] * self.action_dim, self.action_dim).to(new_action.device)
                    indices = torch.arange(state.shape[0] * self.action_dim)
                    column_indices = indices % self.action_dim
                    new_action_one_hot[indices, column_indices] = new_action_flat

                    state_for_one_hot = state.repeat(*([7] + [1] * (state.dim() - 1)))
                    new_q1, new_q2 = self.actorCritic.getQ(state_for_one_hot, new_action_one_hot)
                    
                    assert new_q1.shape == (state.shape[0] * self.action_dim, self.action_dim)
                    assert new_q2.shape == (state.shape[0] * self.action_dim, self.action_dim)

                    action_indices = torch.nonzero(new_action_one_hot, as_tuple=True)[1].view(-1, 1)
                    new_q1 = new_q1.gather(1, action_indices).view(state.shape[0], self.action_dim)
                    new_q2 = new_q2.gather(1, action_indices).view(state.shape[0], self.action_dim)

                else:
                    new_q1, new_q2 = self.actorCritic.getQ(state, new_action)
                
                new_q = new_q1 #torch.min(new_q1,new_q2)
                actor_loss = - new_q.mean()
                
                
                self.actorCritic_o.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), self.grad)
                self.actorCritic_o.step()
                
                self.replaybuffer.reset_max_priority()
                
                ####################
                #soft updata  valuetarget
                ####################
                with torch.no_grad():
                    for target_param,param in zip(self.target.parameters(),self.actorCritic.parameters()):
                        target_param.data.copy_(
                        target_param.data *(1 - self.tau)  + param.data * self.tau
                    )
            
            
            
            ####################
            #updata  Q
            ####################
            
            with torch.no_grad():
                noise = (
                torch.randn_like(action) * self.actionNoise
            ).clamp(-self.noiseClip, self.noiseClip)
                target_a=self.target.getAction(next_state)
                next_action = (target_a + noise).clamp(0, self.max_action)
                # Compute the target Q value
                target_Q1, target_Q2 = self.target.getQ(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                next_q_value = reward + mask * (self.dicount**exp) * target_Q
                
            q1,q2 = self.actorCritic.getQ(state,action)
            
            
            q1_loss = ((q1 - next_q_value)**2).mean()
            q2_loss = ((q2 - next_q_value)**2).mean()
            q_loss = q1_loss + q2_loss
            
            self.actorCritic_o.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actorCritic.parameters(), self.grad)
            self.actorCritic_o.step()
            
            self.scheduler.step()
            
            ####################
            #updata  LAP
            ####################
            
            td_loss = 0.5 * ((q1 - next_q_value).abs() + (q2 - next_q_value).abs())
            priority = td_loss.max(1)[0].clamp(min=self.min_priority).pow(self.LAPalpha)
            self.replaybuffer.update_priority(priority)
            
            
            
            writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
            writer.add_scalar('value_loss', q_loss.item(), global_step=self.learn_step)
            
            if (i+1) % self.num_epch_train == 0:
                process.process_input(self.learn_step, 'learn_step', 'train/')
                process.process_input(actor_loss.item(), 'actor_loss', 'train/')
                process.process_input(q_loss.item(), 'q_loss', 'train/')
                process.process_input(new_q.detach().cpu().numpy(), 'new_q', 'train/')
                process.process_input(target_Q.detach().cpu().numpy(), 'target_Q', 'train/')
                process.process_input(new_action.detach().cpu().numpy(), 'new_action', 'train/')
                
    
    
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
                
                
                
                
            
            
            
            
            
            
            
            
        
        
        
        
