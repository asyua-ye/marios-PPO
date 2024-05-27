import numpy as np
import torch

class RunningMeanStd:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.ones(shape)  # 避免初始除零

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        batch_mean = np.mean(x, axis=0)
        batch_S = np.sum((x - batch_mean) ** 2, axis=0)
        batch_n = x.shape[0]

        if self.n == 0:
            self.mean = batch_mean
            self.S = batch_S
        else:
            total_n = self.n + batch_n
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_n / total_n
            self.S = self.S + batch_S + delta ** 2 * self.n * batch_n / total_n
            self.mean = new_mean

        self.n += batch_n
        self.std = np.sqrt(self.S / self.n)

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
            std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
            return (x - mean) / (std + 1e-8)  # 避免除零
        else:
            return (x - self.mean) / (self.std + 1e-8)  # 避免除零
        
        

        
        


        
        

