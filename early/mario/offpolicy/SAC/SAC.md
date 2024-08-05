## 介绍

SAC,是目前最流行的强化学习算法，特别是在连续动作环境，有大量基于SAC的工作，涵盖各个领域。  

SAC，它的最重要的特点是，软更新，以及通过熵鼓励探索，而在我的mario项目中，我还添加了一些优化，
如下所示：  
1、双Q结构  
2、CNN特征提取归一化
3、奖励缩放
4、同步采样
5、梯度clip
6、学习率衰减

这个版本是基础版，后续会在这个版本上修改。







## 实验讨论

### 原地更新的错误问题

碰到了一个原地更新的错误，具体提示如下：
> Hint: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [4096, 128]], which is output 0 of AsStridedBackward0, is at version 2; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

#### 网络结构说明
网络结构包含一个共享的特征提取层，此层输出分别用于计算 `mean` 和 `std` 以及 `q1`，`q2`。即使是将特征提取层单独使用，并与两个分开的网络结合，仍然会遇到相同的问题。完整代码在test文件夹中的implace1和2。

#### 问题代码
```python
mean, log_std = network(input_data, compute_q=False)
mean_loss = F.mse_loss(mean, mean_target)
q_pred = network(input_data, actions, compute_q=True)
q_loss = F.mse_loss(q_pred, q_target)
optimizer_mean.zero_grad()

mean_loss.backward()
optimizer_mean.step()

optimizer_q.zero_grad()
q_loss.backward()
optimizer_q.step()
```

#### 无问题代码
```python
optimizer_mean.zero_grad()
    
mean, log_std = network(input_data, compute_q=False)
mean_loss = F.mse_loss(mean, mean_target)
 mean_loss.backward()
optimizer_mean.step()
    
optimizer_q.zero_grad()
q_pred = network(input_data, actions, compute_q=True)
q_loss = F.mse_loss(q_pred, q_target)
    
q_loss.backward()
optimizer_q.step()
```

#### 结论
原理就是，在一次训练中，对于有共享的网络结构，进行两次顺序的前向传播，且没有依次清空梯度，就会出现原地错误。



### 一些困惑

SAC，还不如ppo，与我想的不太一样.....或许要调一下参数，只是我还能停留吗？
