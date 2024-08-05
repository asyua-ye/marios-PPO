import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
    
    
class res18(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(res18, self).__init__()
        self.conv1 = nn.Conv2d(state_dim[0], 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, action_dim)
        self.critic_linear = nn.Linear(512, 1)
        
        
        ## finetune?
        self.l1 = nn.Linear(action_dim, 512)
        self.tanh = nn.Tanh()
        self.offset_linear = nn.Linear(512, action_dim)
        
        
        self._initialize_weights()
    
    def forward(self, x, finetune=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        v = self.critic_linear(x)
        
        if finetune:
            logits = logits.detach()
            out = self.tanh(self.l1(logits))
            logits = logits + self.offset_linear(out)
        
        return logits,v

    def get_action(self, state, finetune=False):
        logits,_ = self.forward(state,finetune)
        probabilities = torch.sigmoid(logits)
        action = torch.bernoulli(probabilities).int()
        action_log_probs = torch.log(probabilities * action + (1 - probabilities) * (1 - action))
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
        
        return action, action_log_probs
    
    def getValue(self,state):
        _,value= self.forward(state)
        
        return value
    
    def getLogits(self,state):
        
        logits,_ = self.forward(state)
        
        return logits

    def getLogprob(self, state, old_action,finetune=False):
        logits,_ = self.forward(state,finetune)
        probabilities = torch.sigmoid(logits)
        
        old_logprob = torch.log(probabilities * old_action + (1 - probabilities) * (1 - old_action) + 1e-6)
        old_logprob = old_logprob.sum(dim=-1, keepdim=True)
        
        distentropy = - (probabilities * torch.log(probabilities + 1e-6) + (1 - probabilities) * torch.log(1 - probabilities + 1e-6))
        distentropy = distentropy.mean()
        
        return old_logprob, distentropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.fc.weight, 0.01)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
        nn.init.orthogonal_(self.critic_linear.weight, 1.0)
        if self.fc.bias is not None:
            nn.init.constant_(self.critic_linear.bias, 0)
            
            
class agent(object):
    def __init__(self,state_dim, action_dim,hp) -> None:
        super(agent,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.ActorCritic = res18(state_dim,action_dim).to(self.device)
        self.ActorCritic_optimizer = torch.optim.Adam(self.ActorCritic.parameters(), lr=hp.actor_lr,eps=hp.eps)
        
        lambda_lr = lambda step: 1.0 - step / hp.max_steps if step < int(0.9 *hp.max_steps) else 0.1
        self.scheduler = [LambdaLR(self.ActorCritic_optimizer, lr_lambda=lambda_lr),
                            ]
        
        #PPO
        self.clip = hp.clip
        self.grad = hp.grad
        self.num_epch_train = hp.num_epch_train
        self.entropy = hp.entropy
        self.value_weight = hp.value
        self.actor_weight = hp.actor
        
        #checkpoint
        self.Maxscore = (0.0,0)
        self.learn_step = -1
        
    @torch.no_grad()
    def select_action(self,state,fintune=False):
        
        if state.ndim == 3:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).to(self.device)
        else:
            state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        action,logprob = self.ActorCritic.get_action(state,fintune)
        value = self.ActorCritic.getValue(state)
        action = action.view(-1,self.action_dim).cpu().data.numpy()
        logprob = logprob.view(-1,1).cpu().data.numpy()
        value = value.view(-1,1).cpu().data.numpy()
        
        return action,logprob,value
    
    @torch.no_grad()
    def get_value(self,state):
        
        state = torch.FloatTensor(state.reshape(-1, *state.shape)).squeeze().to(self.device)
        value = self.ActorCritic.getValue(state)
        value = value.view(-1).cpu().data.numpy()
        
        return value
    
    
    def evaluate_actions(self, state,actions):
        
        logprob,dist_entropy = self.ActorCritic.getLogprob(state,actions,True)
        
        
        return logprob, dist_entropy
    
    
    def train(self,sample,process,writer):
        
        self.learn_step += 1
        state,action,old_action_log_probs,returns,advs = sample
        
        action_log_probs, dist_entropy = self.evaluate_actions(state,action)
        
        ratio =  torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advs
        actor_loss = -torch.min(surr1, surr2).sum(dim=-1).mean()
        
        
        actor_loss = self.actor_weight * actor_loss - self.entropy * dist_entropy
        
        
        values = self.ActorCritic.getValue(state)
        value_loss = F.mse_loss(returns, values)
        loss = actor_loss + self.value_weight * value_loss
        
        
        self.ActorCritic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ActorCritic.parameters(), self.grad)
        self.ActorCritic_optimizer.step()
        
        
        for scheduler in self.scheduler:
            scheduler.step()
        
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
                
        if self.learn_step % 1000 ==0:
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
            
    def learn(self, sample, process, writer):
        state, label_action, returns = map(lambda x: x.to(self.device), sample)
        self.learn_step += 1
        
        logits = self.ActorCritic.getLogits(state)
        actor_loss = nn.BCEWithLogitsLoss()(logits, label_action.float())
        
        values = self.ActorCritic.getValue(state)
        value_loss = F.mse_loss(returns, values)
        
        loss = actor_loss + value_loss
        
        self.ActorCritic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ActorCritic.parameters(), self.grad)
        self.ActorCritic_optimizer.step()
        
        for scheduler in self.scheduler:
            scheduler.step()
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
        
        if self.learn_step % self.num_epch_train == 0:
            with torch.no_grad():
                process.process_input(self.learn_step, 'learn_step', 'train/')
                process.process_input(logits.cpu().numpy(), 'logits', 'train/')
                process.process_input(values.cpu().numpy(), 'values', 'train/')
                process.process_input(returns.cpu().numpy(), 'returns', 'train/')
                
    def Imitate(self,sample,writer):
        state, label_action, returns = map(lambda x: x.to(self.device), sample)
        self.learn_step += 1
        
        logits = self.ActorCritic.getLogits(state)
        action_log_probs = F.binary_cross_entropy_with_logits(logits, label_action.float(), reduction='none')
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
        actor_loss = action_log_probs.mean()
        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            # probabilities = torch.clamp(probabilities, 1e-7, 1 - 1e-7)
            new_action = torch.bernoulli(probabilities).int()
            accuracy = (new_action == label_action).float().mean()  # 宽松
            # accuracy = torch.all(new_action == label_action,dim=-1).float().mean()  # 严格
            
        values = self.ActorCritic.getValue(state)
        value_loss = F.mse_loss(returns.view(-1,1), values)
        
        loss = actor_loss + value_loss
        
        self.ActorCritic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ActorCritic.parameters(), self.grad)
        self.ActorCritic_optimizer.step()
        
        for scheduler in self.scheduler:
            scheduler.step()
        
        writer.add_scalar('actor_loss', actor_loss.item(), global_step=self.learn_step)
        writer.add_scalar('value_loss', value_loss.item(), global_step=self.learn_step)
        writer.add_scalar('acc', accuracy.item(), global_step=self.learn_step)
        
        return accuracy.item()
            
            
    def save(self, filename):
        torch.save({
            'ActorCritic_state_dict': self.ActorCritic.state_dict(),
        }, filename + "_ActorCritic")
        
        
    def load(self, filename):
        checkpoint = torch.load(filename + "_ActorCritic")
        self.actor.load_state_dict(checkpoint['ActorCritic_state_dict'])
        
        
    def IsCheckpoint(self,Score):
        """
        Args:
            Score (_type_): (fin,Score),fin代表完成了几次关卡，Score表示总体表现情况

        Returns:
            _type_: 是否是最好的模型
        """
        if self.Maxscore[0]<Score[0]:
            self.Maxscore = Score
            return True
        elif self.Maxscore[0]==Score[0]:
            if self.Maxscore[1]<Score[1]:
                self.Maxscore = Score
                return True
            else:
                return False
        else:
            return False