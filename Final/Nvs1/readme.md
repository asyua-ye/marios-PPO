## 简述 / Overview

本文件主要描述了项目命名由来、各个文件的作用，并详细介绍了一些代码设计。

This document primarily describes the origins of the project's naming conventions, the roles of various files, and provides detailed explanations of certain code designs.

### 命名由来 / Naming Conventions

*   Nvs1: 代表 "N-dimensional vector representation with 1 fixed time frame"，即使用 N 维向量表示动作，且时间帧数固定为 4 帧。  

    Stands for "N-dimensional vector representation with 1 fixed time frame," indicating the use of N-dimensional vectors to represent actions, with a fixed time frame of 4 frames.

*   ppomax: 由于项目中使用了三种稳定训练的方法，故命名为 ppomax。  

    The name "ppomax" reflects the utilization of three methods for stabilizing the training process.

*   res18: 学生模型，主要设计用于学习已经训练好的 32 个教师模型。  

    This denotes the student model, primarily designed to learn from 32 pre-trained teacher models.

### ppomax 文件夹下的文件及说明 / Files and Descriptions within the 'ppomax' Folder

*   **defectmodel:**
    *   包含有缺陷的模型。主要缺陷在于状态归一化没有区分训练和采样评估阶段，一直在收集统计信息，这在训练次数很多的情况下影响很小。 

        Contains models with defects. The primary issue is that state normalization does not differentiate between training and sampling/evaluation phases, continuously collecting statistics. This has minimal impact when the number of training iterations is large.

    *   另外，checkpoint 机制的设计尚未完善。  

        Additionally, the checkpoint mechanism is not well-designed.

    *   尽管如此，32 个关卡的模型都是在这个文件夹中训练的，所以我也保存了下来，但建议在 train 文件夹中训练，效果会更好。

        Nevertheless, models for all 32 levels were trained in this folder, so they are preserved. However, training in the train folder is recommended for better results.

*   **train:**
    *   只在训练阶段收集状态归一化的统计信息，其他阶段不再收集。  

        Collects state normalization statistics only during the training phase, not in other phases.
    *   checkpoint 机制将采样和评估的表现结合起来评估模型的表现，按照 "(通关率, 采样和评估阶段的最小回合分数的和)" 的标准保存模型。   

        The checkpoint mechanism combines sampling and evaluation performance to assess the model, saving models based on the criteria of "(clearance rate, sum of minimum episode scores in sampling and evaluation phases)".

*   **retrain:**
    *   在 train 的基础上增加了 retrain 机制，仅修改了学习率调度以及相应的加载模型的代码。  

        Adds a retraining mechanism on top of train, modifying only the learning rate scheduling and corresponding model loading code.

*   **test:**
    *   32个关卡的训练模型以及训练过程中的所有记录结果。  

        Contains trained models for all 32 levels and all records generated during the training process.

### 其他文件及说明 / Other Files and Descriptions

*   **buffer:** 存储采样过程中的 (state, action, value, done 等) 数据，并具有回放池采样功能。此外，还负责 GAE、优势函数归一化、奖励缩放等。 

    Stores data from the sampling process (state, action, value, done, etc.) and provides replay buffer sampling functionality. It's also responsible for GAE, advantage normalization, and reward scaling.

*   **main:** 负责训练和调整超参数。  

    Handles training and hyperparameter tuning.

*   **getDataset:** 生成数据集，主要用于蒸馏阶段。  

    Generates datasets, primarily for the distillation phase.

*   **getVideo:** 生成视频和 GIF 文件，测试模型的通关率。  

    Creates videos and GIF files to test the model's clearance rate.

*   **PPO:** 模型文件，以及生成动作、状态价值、保存/加载模型等功能。  

    Contains the model files, along with functions for generating actions, state values, saving and loading models.

*   **pre_env:** 对原始的 gym-super-mario 环境的状态进行处理，与 DQN 对 Atari 游戏的处理类似，将图片转换为 84x84 大小，一个动作执行 4 帧并输出最后一帧，累计 4 帧 84x84 的图片作为状态；还包括动作设计和奖励函数设计。  

    Processes the states from the original gym-super-mario environment, similar to how DQN handles Atari games. It converts images to 84x84, executes an action for 4 frames and outputs the last frame, accumulating 4 frames of 84x84 images as one state. It also includes action design and reward function design.

*   **subproc_vec_env:** 使用多进程创建 Mario 环境，主要用于采样。  

    Creates multiple Mario environments using multiprocessing, mainly for sampling.

### 系统组成 / System Composition

整个系统宏观上由训练部分、评估部分和记录部分组成。

The overall system consists of three main components: training, evaluation, and logging.

### 训练部分 / Training

主要介绍 3 个稳定训练的方法以及 8-4 关卡的奖励函数。

This section focuses on three methods for stabilizing training and the reward function for level 8-4.

1.  **状态归一化 / State Normalization**
    *   主要对 CNN 最后一层的输出进行展平后的向量（如 3177 维就有 3177 个均值和方差），每一位进行统计，通过滑动均值和方差更新归一化每一位。  

        Primarily normalizes each element of the flattened vector (e.g., 3177 dimensions have 3177 means and variances) from the output of the last CNN layer. Statistics are collected for each element, and normalization is updated using moving averages and variances.

2.  **奖励缩放 / Reward Scaling**
    *   维持奖励的滑动均值和方差，对每个奖励进行缩放。缩放计算为：奖励值 / 标准差。  

        Maintains moving averages and variances for rewards, scaling each reward. The scaling calculation is: reward value / standard deviation.

3.  **学习率衰减 / Learning Rate Decay**
    *   线性衰减，根据梯度更新步数，从 1.0 衰减到 0.1。  

        Linear decay: Decreases from 1.0 to 0.1 based on the number of gradient update steps.

    *   在 retrain 阶段，学习率会先线性上升，从 0.1 上升到 1.0，然后再从 1.0 余弦衰减到 0.1。这样设计主要是我没有保存critic网络的参数(其中CNN模块是共享的)，需要一个预热期训练critic网络。  

        In the retraining phase, the learning rate first linearly increases from 0.1 to 1.0, then cosine decays from 1.0 to 0.1.The main reason for this design is that I didn't save the parameters of the Critic network (where the CNN module is shared) and needed a warm-up period to train the Critic network.  

4.  **8-4 关卡奖励函数 / 8-4 Level Reward Function**
    *   当 agent 进入某些错误路线时会受到惩罚。  

        Penalizes the agent when it enters certain incorrect paths.

### 评估部分 / Evaluation

每次训练完成后，进行一定次数的评估。

After each training session, the model is evaluated a certain number of times.

### 记录部分 / Logging
可以查看 test 文件夹中的文件，主要包括超参数、整个训练过程中的控制台输出、训练过程中的中间值（但只有总共训练回合数次的记录，如总共 1000 次训练，就只有 1000 行记录），以及 TensorBoard 文件记录整个训练过程中的 actor loss 和 value loss（因为基本不看所以没有上传）。   

You can examine the files within the test folder, which primarily include hyperparameters, console output during the entire training process, intermediate values during training (but only records for the total number of training episodes, e.g., if there are 1000 training episodes in total, there will be only 1000 lines of records), and TensorBoard files recording the actor loss and value loss throughout the training process (not uploaded as they are rarely viewed).


## 一些经验

关卡选择: 我在 8-4 和 8-1 这两个关卡上花费了最多的调参时间。如果您想实验其他强化学习算法但时间有限，可以重点关注这两个关卡。另外，8-4的训练极其不稳定，即使是同样的超参数，可能这一次得到了很好的表现，但是下一次就不能保证得到相同的表现(固定了随机种子的情况下)。  

Level Selection: I spent the most time tuning hyperparameters on levels 8-4 and 8-1. If you're interested in experimenting with other reinforcement learning algorithms but have limited time, you may want to focus on these two levels.In addition, 8-4 training is extremely unstable; even with the same hyperparameters, one may get good performance one time, but not be guaranteed to get the same performance the next time (with the random seed fixed).  


实验数据: 我保存了在 Mario 项目中所有的尝试，大约有 10G（包括各种代码版本和调参结果）。如果您想查看这些数据，可以联系我，我会发给您。虽然我觉得可能用处不大，但对我来说主要是一种纪念。  

Experiment Data: I have saved all the data from my experiments on the Mario project, which amounts to about 10GB (including various code versions and hyperparameter tuning results). If you'd like to take a look, feel free to contact me, and I'll send it over. Although I don't think it will be of much use, it serves as a memento for me.





