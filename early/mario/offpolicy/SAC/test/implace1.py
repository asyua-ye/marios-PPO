import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 128)  # 假设输入为8x8的图像

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(QNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.q_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        q_value = self.q_head(features)
        return q_value

# 定义mean网络
class MeanNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(MeanNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.mean_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_value = self.mean_head(features)
        return mean_value

# 初始化网络
feature_extractor = FeatureExtractor()
q_network = QNetwork(feature_extractor)
mean_network = MeanNetwork(feature_extractor)

# 定义优化器
optimizer_q = optim.Adam(q_network.parameters(), lr=0.001)
optimizer_mean = optim.Adam(mean_network.parameters(), lr=0.001)

# 模拟数据
batch_size = 16
input_data = torch.randn(batch_size, 1, 8, 8)
q_target = torch.randn(batch_size, 1)
mean_target = torch.randn(batch_size, 1)

# 训练步骤
for epoch in range(10):  # 假设进行10个epoch的训练

    
    
    
    mean_pred = mean_network(input_data)
    mean_loss = F.mse_loss(mean_pred, mean_target)
    
    q_pred = q_network(input_data)
    q_loss = F.mse_loss(q_pred, q_target)
    
    optimizer_mean.zero_grad()
    mean_loss.backward()
    optimizer_mean.step()

    
    optimizer_q.zero_grad()
    q_loss.backward()
    optimizer_q.step()
    
    print(f"Epoch {epoch + 1}: Q Loss: {q_loss.item()}, Mean Loss: {mean_loss.item()}")

