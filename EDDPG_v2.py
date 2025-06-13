import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import math
from sklearn import preprocessing




# 环境参数配置
CONFIG = {
    "buffer_size": 10000,
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 0.001,
    "ou_theta": 0.15,
    "ou_sigma": 0.3,
    "lr_actor": 1e-4,
    "lr_critic": 1e-4,
    "num_episodes": 500,
    "state_size": 30,
    "action_size": 30,
    "log_interval": 50
}



class EnhancedEnv:
    def __init__(self, data_path):
        self.load_data(data_path)
        self.bandwidth = 100 * 2 ** 20 / 8  # 100Mbps -> MB/s
        self.edge_capacity = 10. * 10 ** 9  # 10 GHz
        #unit_energy_consumption = 1.42 * 10 ** -7  # 本地能量消耗因子
        #unit_time_consumption = 4.75 * 10 ** -7  # 本地时间消耗因子

    def load_data(self, path):
        df = pd.read_csv(path)
        self.task_data = df['takesize'].values
        self.total_tasks = len(self.task_data)

    def get_state(self, pointer):
        start = pointer % self.total_tasks
        end = (start + total_task) % self.total_tasks
        if end > start:
            return self.task_data[start:end]
        return np.concatenate([self.task_data[start:], self.task_data[:end]])

    def calculate_utility(self, state, action):


        if(time_step <260 and time_step>=250 ):beta=1
        elif(time_step <410 and time_step>=400 ):beta=1
        else:beta=1


        utilities = []
        for i in range(10):  # 10 devices
            device_utils = []
            for j in range(num_tasks):  # 3 tasks per device
                idx = i * num_tasks + j
                data_size = state[idx]
                offload_ratio = action[idx]

                # 本地计算消耗
                unit_energy_consumption = 1.42 * 10 ** -7  # 本地能量消耗因子
                unit_time_consumption = 4.75 * 10 ** -7  # 本地时间消耗因子
                local_data = data_size * (1 - offload_ratio)* 8 * 2 ** 20
                energy_local = local_data * unit_energy_consumption  # 本地能耗
                time_local = local_data * unit_time_consumption # 本地时延


                # 边缘计算消耗
                a = 1.5 * 10 ** -7  # 云端额外能量消耗因子
                APP = 1900.  # 一个常数，用于计算云端处理时间
                edge_data = data_size * offload_ratio* 8 * 2 ** 20
                tx_time = edge_data / self.bandwidth
                compute_time = APP * edge_data / self.edge_capacity
                energy_edge = edge_data * a +edge_data * unit_energy_consumption  # 边缘能耗系数

                # 总效用计算
                total_energy = energy_local + energy_edge
                total_time = max(time_local, tx_time + compute_time)
                utility = total_energy + beta * total_time
                device_utils.append(utility)

            utilities.extend(device_utils)
        return np.array(utilities)


class AttentionActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        output=(self.net(state) + 1) / 2
        #print(output)
        return output # 输出范围[0,1]


class AttentionCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.q_net = nn.Sequential(
            nn.Linear(256 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state_feat = self.state_encoder(state)
        return self.q_net(torch.cat([state_feat, action], dim=1))


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states))
        )


class AttentionDDPG:
    def __init__(self, config):
        self.config = config

        # 初始化网络
        self.actor = AttentionActor(config["state_size"], config["action_size"])
        self.critic = AttentionCritic(config["state_size"], config["action_size"])
        self.target_actor = AttentionActor(config["state_size"], config["action_size"])
        self.target_critic = AttentionCritic(config["state_size"], config["action_size"])
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config["lr_actor"])
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config["lr_critic"])

        # 经验回放
        self.memory = ExperienceReplay(config["buffer_size"])
        self.noise = OUNoise(config["action_size"], theta=config["ou_theta"], sigma=config["ou_sigma"])

        # 训练日志
        self.log = {
            "episode": [],
            "total_utility": [],
            "action_change": [],
            "alpha_values": [],
            "critic_loss": [],  # 新增
            "actor_loss": []  # 新增
        }

    def hard_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)

    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.config["tau"] * s_param.data + (1 - self.config["tau"]) * t_param.data)

    def get_action(self, state, episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # 生成基础动作
        with torch.no_grad():
            base_action = self.actor(state_tensor).squeeze().numpy()

        # 添加探索噪声
        #print("初始：",base_action)
        #print("噪声：",self.noise())

        action1=base_action + self.noise()
        action = np.tanh(action1)
        noisy_action=(action+1)/2

        #noisy_action = np.clip(base_action + self.noise(), 0, 1)
        #print("网络输出：",noisy_action)

        # 注意力融合
        fused_action = self.attention_fusion(state, noisy_action, episode)

        return {
            "base": base_action,
            "noisy": noisy_action,
            #"final": fused_action,
            #消融实验设置："final": based_action,
            "final": fused_action,
            "alpha": self.get_alpha(episode)
        }

    def get_alpha(self, episode):
        """动态融合权重"""
        l=0.01
        w_t0=0.5
        alpha= w_t0 * (math.e ** (-l*episode))
        return alpha

    def attention_fusion(self, current_state, current_action, episode):
        """两阶段注意力融合"""
        # 阶段一：按奖励筛选
        rewards = [exp[2] for exp in self.memory.buffer]
        if len(rewards) < 30:
            return current_action

        #reward_threshold = np.percentile(rewards, 80)
        reward_threshold = np.percentile(rewards, 80)
        S1 = [exp for exp in self.memory.buffer if exp[2] >= reward_threshold]
        #print("HHH",S1)
        # 阶段二：按状态相关性筛选
        current_flat = current_state.flatten()
        similarities = []
        samples = []
        #for exp in S1:
        for exp in S1:
            hist_state = exp[0].flatten()
            min_len = min(len(current_flat), len(hist_state))

            try:
                corr = np.corrcoef(current_flat[:min_len], hist_state[:min_len])[0, 1] + 0.7
                #if corr > 0.8 and not np.isnan(corr):
                if corr > 0.8 and not np.isnan(corr):
                    similarities.append(corr)
                    #similarities.append(0)
                    samples.append(exp)
            except:
                continue


        #print(s_max)

        #print(similarities)
        if not samples:
            return current_action

        # 注意力权重计算
        weights = F.softmax(torch.tensor(similarities) / 0.2, dim=0).numpy()

        # 动作融合
        fused = np.zeros_like(current_action)
        for w, exp in zip(weights, samples):
            fused += w * exp[1]

        alpha = self.get_alpha(episode)
        return alpha * current_action + (1 - alpha) * fused

    def train_step(self):
        batch = self.memory.sample(self.config["batch_size"])
        if batch is None:
            return

        states, actions, rewards, next_states = batch

        # Critic更新
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            target_q = rewards + self.config["gamma"] * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor更新
        policy_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # 目标网络更新
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return critic_loss.item(), policy_loss.item()

    def save_logs(self, path):
        df = pd.DataFrame(self.log)
        df.to_csv(os.path.join(path, "training_log.csv"), index=False)

        plt.figure(figsize=(18, 6))

        # 原始图表
        plt.subplot(131)
        plt.plot(df["episode"], df["total_utility"])
        plt.title("Total Utility per Episode")

        plt.subplot(132)
        plt.plot(df["episode"], df["action_change"])
        plt.title("Action Adjustment Magnitude")

        # 新增损失图表
        plt.subplot(133)
        plt.plot(df["episode"], df["critic_loss"], label="Critic Loss")
        plt.plot(df["episode"], df["actor_loss"], label="Actor Loss")
        plt.legend()
        plt.title("Training Losses")

        plt.tight_layout()
        plt.savefig(os.path.join(path, "training_curve.png"))
        plt.close()


class OUNoise:
    def __init__(self, dim, theta=CONFIG["ou_theta"], sigma=CONFIG["ou_sigma"]):
        self.theta = theta
        self.sigma = sigma
        self.min_sigma = 0
        self.sigma_decay = 0.99
        self.pa=0.1
        self.dim = dim
        self.reset()


    def reset(self):
        self.first = np.ones(self.dim) * 0.5

    def __call__(self):
        dx = self.theta * (0.5 - self.first) + self.sigma * np.random.randn(self.dim)
        self.first += dx
        self.first*=(self.sigma_decay**time_step)
        #self.first*=(self.sigma_decay*(math.e**(-self.pa*time_step)))
        #print(self.sigma_decay**time_step)
        #self.sigma=self.sigma_decay*self.sigma_decay
       # print("sigma值为：",self.sigma_decay)
        return self.first



def main():
    # 初始化组件
    config = CONFIG
    env = EnhancedEnv("./data/test2.csv")
    agent = AttentionDDPG(config)
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    EDDPG=0.
    EDDPG_sum=[]
    AC_loss=[]
    CR_loss=[]
    global time_step

    global num_devices  # 设备数量
    num_devices = 10
    global num_tasks  # 每个设备的任务数量
    num_tasks = 3
    global total_task
    total_task= num_tasks * num_devices
    # 训练循环
    for ep in range(config["num_episodes"]):
        time_step=ep
        # 获取当前状态
        pointer = ep * total_task % env.total_tasks
        state = env.get_state(pointer)

        # 生成动作
        action_info = agent.get_action(state, ep)

        # 计算效用
        utilities = env.calculate_utility(state, action_info["final"])
        total_utility = np.sum(utilities)
        EDDPG+=total_utility
        EDDPG_sum.append(total_utility)


        # 存储经验
        reward = -total_utility  # 假设奖励是效用的负值
        next_pointer = (pointer + total_task) % env.total_tasks
        next_state = env.get_state(next_pointer)
        agent.memory.add((state, action_info["final"], reward, next_state))

        # 训练模型
        train_loss = agent.train_step()

        # 初始化默认损失值
        critic_loss = 0.0
        actor_loss = 0.0

        # 如果有实际训练发生
        if train_loss is not None:
            critic_loss, actor_loss = train_loss

        # 记录日志
        action_change = np.mean(np.abs(action_info["final"] - action_info["base"]))
        agent.log["episode"].append(ep)
        agent.log["total_utility"].append(total_utility)
        agent.log["action_change"].append(action_change)
        agent.log["alpha_values"].append(action_info["alpha"])
        agent.log["critic_loss"].append(critic_loss)  # 记录critic损失
        agent.log["actor_loss"].append(actor_loss)     # 记录actor损失
        CR_loss.append(critic_loss)
        AC_loss.append(actor_loss)

        # 定期输出
        if ep % config["log_interval"] == 0:
            print(f"Episode {ep + 1}/{config['num_episodes']}")
            print(f"Utility: {total_utility:.1f} | Action Change: {action_change:.3f}")
            print(f"Base Action: {np.mean(action_info['base']):.3f}±{np.std(action_info['base']):.3f}")
            print(f"Final Action: {np.mean(action_info['final']):.3f}±{np.std(action_info['final']):.3f}")
            if train_loss:
                print(f"Critic Loss: {train_loss[0]:.4f} | Actor Loss: {train_loss[1]:.4f}")
            print("-" * 50)
    loss_data = np.column_stack((AC_loss, CR_loss))
    #np.savetxt("./Data_V2/exp_sum/EDDPG_data.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_sum/EDDPG_data.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_ur/EDDPG_data_1_0.5_2.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_ur/EDDPG_data_1.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_lr/EDDPG_data_0.5.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_bs/EDDPG_experience_1024.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")
    #.savetxt("./Data_V2/exp_lr_loss/EDDPG_loss_11.csv", loss_data, delimiter=",", header="EDDPG_AC,EDDPG_CR", comments='', fmt="%.2f")
    #np.savetxt("./Data_V2/exp_Y_N/N_Cost.csv", EDDPG_sum, delimiter=",", header="EDDPG_Cost", comments='', fmt="%.2f")

    print(EDDPG)
    #print(EDDPG_sum)
    # 保存结果
    agent.save_logs(log_dir)
    torch.save(agent.actor.state_dict(), os.path.join(log_dir, "final_actor.pth"))


if __name__ == "__main__":
    main()