import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt

# 定义环境类
class Env:
    def __init__(self, num_devices, num_tasks):
        # 初始化环境参数，包括带宽、边缘处理速率、能量和时间消耗等
        self.bandwidth = 100 * 2 ** 20 / 8  # 带宽，单位为MB/s
        self.edge_processing_rate = 10. * 10 ** 9  # 边缘处理速率
        self.unit_energy_consumption = 1.42 * 10 ** -7  # 本地能量消耗因子
        self.unit_time_consumption = 4.75 * 10 ** -7  # 本地时间消耗因子
        self.fixed_cloud_energy_cost = 1.5 * 10 ** -7  # 云端固定能耗
        self.a = 1.5 * 10 ** -7  # 云端额外能量消耗因子
        self.APP = 1900.  # 一个常数，用于计算云端处理时间

    # 每步的计算，返回效用值
    def step(self, data_size, offload_ratio):
        local_data_size = data_size * (1 - offload_ratio) * 8 * 2 ** 20  # 本地处理数据大小
        cloud_data_size = data_size * offload_ratio * 8 * 2 ** 20  # 云端处理数据大小

        # 计算本地能耗和总时间
        local_energy_consumption = local_data_size * self.unit_energy_consumption
        total_local_time = local_data_size * self.unit_time_consumption

        # 计算云端传输延迟和处理时间
        cloud_transmission_delay = cloud_data_size / self.bandwidth
        edge_processing_time = self.APP * cloud_data_size / self.edge_processing_rate
        cloud_processing_time = edge_processing_time + cloud_transmission_delay

        # 计算云端能耗和总能耗
        cloud_energy_consumption = cloud_data_size * self.fixed_cloud_energy_cost + self.a * cloud_data_size
        total_energy_consumption = local_energy_consumption + cloud_energy_consumption

        # 效用函数，beta为平衡因子
        beta = 1
        utility = total_energy_consumption + beta * np.maximum(total_local_time, cloud_processing_time)

        return utility

# 定义DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, net_num):
        self.state_size = state_size  # 输入的状态大小（任务数）
        self.action_size = action_size  # 动作空间大小（卸载比例的离散化数量）
        self.net_num = net_num  # 控制网络数量

        self.net = [state_size, 128, 128, action_size * 30]  # 定义神经网络结构，最后输出为每个任务的卸载比例

        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size=10000, batch_size=32)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        self.learning_rate = 0.001  # 学习率
        self.losses = []  # 存储损失值
        self._build_net()  # 构建网络
        self.optimizer = optim.Adam(self.memory0.parameters(), lr=self.learning_rate)  # 使用Adam优化器
        self.update_target_model()  # 更新目标模型

        # 定义离散化的卸载比例
        self.offload_ratios = np.linspace(0, 1, action_size)  # 动作空间离散化

    def _build_net(self):
        # 动态创建多个神经网络
        for i in range(self.net_num):
            setattr(self, f'memory{i}', nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),  # 输入到隐藏层
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),  # 隐藏层
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3])  # 隐藏层到输出
            ))

    # 动作选择函数
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 随机选择动作（探索）
            return np.random.choice(self.offload_ratios, size=(30,))  # 为30个任务随机选择卸载比例

        state = torch.FloatTensor(state).unsqueeze(0)  # 将状态转为Tensor并增加一个批次维度
        with torch.no_grad():
            q_values = getattr(self, f'memory0')(state)  # 获取Q值
            q_values = q_values.view(-1, 30, self.action_size)  # 调整形状以适应任务和动作空间

        actions = torch.argmax(q_values, dim=2).numpy()  # 选择每个任务Q值最大的动作
        return self.offload_ratios[actions].squeeze()  # 返回卸载比例

    # 更新目标模型
    def update_target_model(self):
        # 将当前网络的权重复制到目标模型
        for i in range(self.net_num):
            getattr(self, f'memory{i}').load_state_dict(getattr(self, f'memory{i}').state_dict())

    # 训练模型
    def train(self, state, action, reward, next_state, done):
        # 如果不是终止状态，则计算目标值
        target = reward
        if not done:
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward + self.gamma * torch.max(
                getattr(self, f'memory0')(next_state).view(-1, 30, self.action_size)).item()

        # 获取当前状态的Q值，并调整形状
        state = torch.FloatTensor(state).unsqueeze(0)
        target_f = getattr(self, f'memory0')(state).detach().numpy().reshape(1, 30, self.action_size)
        for i in range(30):
            target_f[0, i, action[i]] = target[i]  # 更新每个任务特定动作的目标值

        # 执行梯度下降
        self.optimizer.zero_grad()
        output = getattr(self, f'memory0')(state).view(-1, 30, self.action_size)
        loss = nn.MSELoss()(output, torch.FloatTensor(target_f))
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())  # 保存损失

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 定义ReplayBuffer类，用于经验回放
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)  # 经验池，最大长度为buffer_size
        self.batch_size = batch_size  # 每次采样的数量

    def add(self, experience):
        # 向经验池中添加经验
        self.buffer.append(experience)

    def sample(self):
        # 随机采样经验
        experiences = random.sample(self.buffer, k=self.batch_size)
        states, actions, rewards, next_states, task_indices = zip(*experiences)
        return np.vstack(states), np.array(actions,dtype=int), np.array(rewards), np.vstack(next_states), np.array(task_indices)

    def __len__(self):
        return len(self.buffer)  # 返回当前经验池大小

history_utilities=[]
# 主函数
def main():
    num_devices = 10  # 设备数量
    num_tasks = 3  # 每个设备的任务数量
    dqnet_num = 3  # 使用的网络数量
    task_count = num_devices * num_tasks  # 计算总任务数量
    DQNSum=0.
    LocalSum=0.
    EdgeSum=0.
    RandomSum=0.
    MinSum=0.

    env = Env(num_devices, num_tasks)  # 创建环境
    agent = DQNAgent(state_size=task_count, action_size=21, net_num=dqnet_num)  # 创建DQN智能体

    num_episodes = 500  # 训练的轮数

    # 读取CSV数据，包含任务的大小
    #data = pd.read_csv('./data/MUMT_data_3x3.csv')
    data = pd.read_csv('./data/test2.csv')
    task_sizes = data['takesize'].values  # 提取任务大小列
    total_tasks = len(task_sizes)  # 总任务数量
    tasks_taken = 0  # 用于记录已经提取了多少任务

    #episode_DDPG=pd.read_csv('Data_V2/exp_sum/DDPG_data.csv')['D_Cost']
    #print(np.cumsum(episode_DDPG))
    #实验1 每个时隙的成本总和

    episode_Edge = []
    episode_Local = []
    episode_Random = []
    episode_DQN = []

    for e in range(num_episodes):
        # 每次从CSV中提取30个任务的大小
        if tasks_taken + 30 <= total_tasks:
            state = task_sizes[tasks_taken:tasks_taken + 30]
        else:
            remaining = total_tasks - tasks_taken
            state = np.concatenate((task_sizes[tasks_taken:], task_sizes[:30 - remaining]))

        tasks_taken = (tasks_taken + 30) % total_tasks  # 更新已提取任务数

        # 获取代理的卸载比例决策
        offload_ratios = agent.act(state)

        utilities = []  # 存储效用


        #实验1
        Edge=0.
        Local=0.
        Random=0.
        DQN=0.

        # 根据历史效用设置奖励阈值
        if len(history_utilities) > 300:
            mean_utility = np.mean(history_utilities)
            std_utility = np.std(history_utilities)
            high_threshold = mean_utility + 3 * std_utility
            middle_threshold = mean_utility + 2 * std_utility
            low_threshold = mean_utility + std_utility
        else:
            high_threshold, middle_threshold, low_threshold = 1000000, 1000000, 1000000

        # 计算每个设备的效用
        for device_index in range(num_devices):
            device_utility = 0  # 初始化设备效用
            device_utilitynum = []  # 存储单个设备的任务效用

            for task_index in range(num_tasks):
                # 计算每个任务的效用
                utility = env.step(data_size=state[3 * device_index + task_index],
                                   offload_ratio=offload_ratios[3 * device_index + task_index])

                DQN+=utility
                DQNSum+=utility

                Edge+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=1)
                EdgeSum+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=1)

                LocalSum+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=0)
                Local+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=0)

                RandomSum+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=random.uniform(0, 1))
                Random+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=random.uniform(0, 1))

                MinSum+=env.step(data_size=state[3 * device_index + task_index],offload_ratio=0.64)

                device_utilitynum.append(utility)
                device_utility += utility

            history_utilities.append(device_utility)  # 记录设备的总效用

            # 根据效用阈值调整奖励
            """
            
            1.若用户平均值高于阈值   是否是每个用户加惩罚  
            2.送入经验回放池的是用户整体，还是单个任务
            
            """
            if device_utility > high_threshold:
                device_utilitynum = 1.15 * np.array(device_utilitynum)  # 高奖励
            elif device_utility > middle_threshold:
                device_utilitynum = 1.1 * np.array(device_utilitynum)  # 中等奖励
            elif device_utility > low_threshold:
                device_utilitynum = 1.05 * np.array(device_utilitynum)  # 低奖励

            for utility in device_utilitynum:
                utilities.append(-utility)  # 负的效用作为奖励

        episode_Edge.append(Edge/10.0)
        episode_Local.append((Local/10.0))
        episode_Random.append(Random/10.0)
        episode_DQN.append(DQN/10.0)



        reward = np.array(utilities)  # 将效用转为奖励

        next_state = state.copy()  # 假设状态不变
        done = False  # 终止标志

        agent.memory.add((state, offload_ratios, reward, next_state, done))  # 添加经验到缓冲区

        # 当经验池足够大时进行训练
        if len(agent.memory) > agent.memory.batch_size:
            experiences = agent.memory.sample()
            for state_, action_, reward_, next_state_, done_ in zip(*experiences):
                agent.train(state_, action_, reward_, next_state_, done_)

        print('输入的30个任务大小为：' )
        print(state)
        print('决策的30个卸载比例为:')
        print(offload_ratios)

        # 每隔10轮更新目标模型
        if e % 10 == 0:
            agent.update_target_model()



    #np.savetxt("./Data_V2/exp_sum/Local_data.csv", episode_Local, delimiter=",", header="Local_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_sum/Edge_data.csv", episode_Edge, delimiter=",", header="Edge_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_sum/Random_data.csv", episode_Random, delimiter=",", header="Random_Cost", comments='',fmt="%.2f")
    #np.savetxt("./Data_V2/exp_sum/DQN_data.csv", episode_DQN, delimiter=",", header="DQN_Cost", comments='', fmt="%.2f")


    # 绘制损失曲线
    """plt.plot(agent.losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.show()"""


    print('DQN最后得出的总效用：',DQNSum)
    print('Local最后得出的总效用：',LocalSum)
    print('Edge最后得出的总效用：',EdgeSum)
    print('随机卸载最后得出的总效用：',RandomSum)
    print('最小总效用：',MinSum)

if __name__ == "__main__":
    main()
