import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义合并后的网络
class CombinedNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=1):
        super(CombinedNetwork, self).__init__()
        # 特征提取层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 128)  # 假设输入为8x8的图像

        # Q网络头
        self.q_head1 = nn.Linear(128 + num_actions, 64)
        self.q_head2 = nn.Linear(64, 1)

        # mean网络头
        self.mean_head = nn.Linear(128, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x, action=None, compute_q=True):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc(x))

        if compute_q and action is not None:
            # 计算Q值
            sa = torch.cat([features, action], 1)
            q1 = F.relu(self.q_head1(sa))
            q_value = self.q_head2(q1)
            return q_value
        else:
            # 计算mean
            mean = self.mean_head(features)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std

# 初始化网络
network = CombinedNetwork(input_channels=1, num_actions=1)

# 定义优化器
optimizer_q = optim.Adam([
    {'params': network.conv1.parameters()},
    {'params': network.conv2.parameters()},
    {'params': network.fc.parameters()},
    {'params': network.q_head1.parameters()},
    {'params': network.q_head2.parameters()},
], lr=0.001)

optimizer_mean = optim.Adam([
    {'params': network.conv1.parameters(), 'lr': 0.0001},
    {'params': network.conv2.parameters(), 'lr': 0.0001},
    {'params': network.fc.parameters(), 'lr': 0.0001},
    {'params': network.mean_head.parameters()},
    {'params': network.log_std}
], lr=0.001)

# 模拟数据
batch_size = 16
input_data = torch.randn(batch_size, 1, 8, 8)
actions = torch.randn(batch_size, 1)
q_target = torch.randn(batch_size, 1)
mean_target = torch.randn(batch_size, 1)

# 训练步骤
for epoch in range(10):  # 假设进行10个epoch的训练
    # 计算Q值损失并更新Q网络
    optimizer_q.zero_grad()
    q_pred = network(input_data, actions, compute_q=True)
    q_loss = F.mse_loss(q_pred, q_target)
    q_loss.backward()
    optimizer_q.step()

    # 计算mean损失并更新mean网络
    optimizer_mean.zero_grad()
    mean, log_std = network(input_data, compute_q=False)
    mean_loss = F.mse_loss(mean, mean_target)
    mean_loss.backward()
    optimizer_mean.step()

    print(f"Epoch {epoch + 1}: Q Loss: {q_loss.item()}, Mean Loss: {mean_loss.item()}")

