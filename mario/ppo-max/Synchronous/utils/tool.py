import numpy as np
import torch
import pandas as pd
import os
from collections import OrderedDict

def log_and_print(text_list, message):
    print(message)
    text_list.append(message)

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
        
        
        
class DataProcessor:
    def __init__(self, file_path=None):
        """
        Args:
            file_name (str, optional): _description_. Defaults to "output.xlsx".
            
        主要是记录每次训练周期的第一次训练到评估中的数据(loss,新计算的v值,时间等)，就像RLkit一样，当然也可以统计多次(不过文件可能会很大..)。
        """
        self.data = OrderedDict()
        self.file_name = (file_path if file_path else "") + "output.xlsx"

    def process_input(self, data, name, prefix=""):
        # 检测数据形状并执行相应的统计操作
        data = np.squeeze(data)  # 去除单维度条目
        if data.size == 1:
            # 对于单个数值，直接记录该值
            self.data[f'{prefix}{name}'] = data.item()  # 使用 .item() 转换 numpy 标量为 Python 标量
        elif data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            # 对于一维数组
            stats = {
                'max': np.max(data),
                'min': np.min(data),
                'mean': np.mean(data),
                'std': np.std(data)
            }
            for stat_name, value in stats.items():
                self.data[f'{prefix}{name}_{stat_name}'] = value
        elif data.ndim == 2:
            # 对于二维数组，计算每列的统计值
            # for i in range(data.shape[1]):
            #     column_data = data[:, i]
            #     self.process_input(column_data, name=f"{name}_col{i}", prefix=prefix)
            stats = {
                'max': np.max(data),
                'min': np.min(data),
                'mean': np.mean(data),
                'std': np.std(data)
            }
            for stat_name, value in stats.items():
                self.data[f'{prefix}{name}_{stat_name}'] = value

    def write_to_excel(self):
        # 检查文件是否存在
        if not os.path.isfile(self.file_name):
            # 如果文件不存在，创建新的 DataFrame 并写入
            df = pd.DataFrame([self.data])
            df.to_excel(self.file_name, index=False)
        else:
            # 如果文件存在，追加数据
            with pd.ExcelWriter(self.file_name, mode='a', if_sheet_exists='overlay') as writer:
                df = pd.DataFrame([self.data])
                if 'Sheet1' in writer.sheets:
                    startrow = writer.sheets['Sheet1'].max_row
                else:
                    startrow = 0
                df.to_excel(writer, index=False, startrow=startrow, header=not os.path.exists(self.file_name))
        
        

        
        


        
        

