import math
import random
import csv
import os
from collections import deque, namedtuple, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Config (已統一，確保公平比較)
# =========================
@dataclass
class Config:
    num_ues: int = 4
    num_channels: int = 4

    max_queue_len: int = 200
    history_len: int = 4

    episode_length: int = 200
    num_episodes: int = 100

    arrival_lambda: float = 0.3  # 確保是 float 格式

    # 共同參數
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    seed: int = 42

    # Full DQN 專屬參數
    batch_size: int = 64
    replay_size: int = 20000
    min_replay_size: int = 500
    target_update_freq: int = 200
    hidden_dim: int = 128

    # Reward (完全一致)
    r_success: float = 5.0
    r_collision: float = -3.0
    r_idle_empty: float = 0.2
    r_idle_backlog: float = -0.5
    r_queue_penalty: float = -0.05
    r_delay_penalty: float = -0.02
    r_tx_attempt: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_ue_details_every_episode: bool = False

    # 新增：動態產生實驗名稱標籤
    def get_exp_suffix(self) -> str:
        return f"ch{self.num_channels}_n{self.num_ues}_arr{self.arrival_lambda}"

# =========================
# Utility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_exponential_slots(lam: float) -> int:
    x = np.random.exponential(scale=1.0 / lam)
    return max(1, int(math.ceil(x)))

# =========================
# Replay Buffer
# =========================
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([b.state for b in batch], dtype=np.float32)
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.array([b.next_state for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# =========================
# Q Network
# =========================
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Agents (Full DQN & Lightweight RL)
# =========================
class DQNAgent:
    """Full DQN: Replay Buffer, Target Network, hidden_dim=128"""
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.replay_size)

        self.learn_step = 0
        self.total_steps = 0

    def get_epsilon(self):
        ratio = min(1.0, self.total_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, state, training=True):
        self.total_steps += 1
        epsilon = self.get_epsilon() if training else 0.0

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.cfg.min_replay_size:
            return None
        if len(self.replay_buffer) < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.cfg.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

class LightweightDQNAgent:
    """Lightweight RL: No Replay Buffer, No Target Net, hidden_dim=32, Online update"""
    def __init__(self, state_dim: int, action_dim: int, cfg: Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # hidden_dim 強制改為 32
        self.q_net = QNetwork(state_dim, action_dim, 32).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        self.total_steps = 0

    def get_epsilon(self):
        ratio = min(1.0, self.total_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, state, training=True):
        self.total_steps += 1
        epsilon = self.get_epsilon() if training else 0.0

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())
    
    def train_step_online(self, state, action, reward, next_state, done):
        """Online update without buffer"""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done_t = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        q_value = self.q_net(state_t).gather(1, action_t)

        with torch.no_grad():
            # No target net, bootstrap from current q_net
            next_q = self.q_net(next_state_t).max(dim=1, keepdim=True)[0]
            target = reward_t + self.cfg.gamma * next_q * (1.0 - done_t)

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

# =========================
# UE & Baseline & Utils
# =========================
class UE:
    def __init__(self, ue_id: int, cfg: Config):
        self.id = ue_id
        self.cfg = cfg
        # 製造不平衡流量
        if self.id < (cfg.num_ues // 2):
            self.arrival_lambda = cfg.arrival_lambda * 2.0
        else:
            self.arrival_lambda = cfg.arrival_lambda * 0.2
        self.reset()

    def reset(self):
        self.queue_len = random.randint(0, 3)
        self.hol_delay = 0
        self.next_arrival_timer = sample_exponential_slots(self.arrival_lambda)

        self.last_access_action = 0
        self.last_channel_action = 0
        self.last_result = 0

        self.collision_hist = deque([0] * self.cfg.history_len, maxlen=self.cfg.history_len)
        self.success_hist = deque([0] * self.cfg.history_len, maxlen=self.cfg.history_len)
        self.queue_hist = deque([0] * self.cfg.history_len, maxlen=self.cfg.history_len)
        self.delay_hist = deque([0] * self.cfg.history_len, maxlen=self.cfg.history_len)

        self.episode_arrivals = 0
        self.episode_tx = 0
        self.episode_listen = 0
        self.episode_success = 0
        self.episode_collision = 0

    def arrival_step(self):
        self.next_arrival_timer -= 1
        if self.next_arrival_timer <= 0:
            if self.queue_len < self.cfg.max_queue_len:
                self.queue_len += 1
            self.episode_arrivals += 1
            self.next_arrival_timer = sample_exponential_slots(self.arrival_lambda)

    def update_delay(self):
        if self.queue_len > 0:
            self.hol_delay += 1
        else:
            self.hol_delay = 0

    def get_state(self):
        return np.array([
            self.queue_len / self.cfg.max_queue_len,
            self.hol_delay / 20.0,
            min(self.next_arrival_timer, 20) / 20.0,
            float(self.last_access_action),
            self.last_channel_action / max(1, self.cfg.num_channels),
            self.last_result / 2.0,
            np.mean(self.collision_hist),
            np.mean(self.success_hist),
            np.mean(self.queue_hist) / self.cfg.max_queue_len,
            np.mean(self.delay_hist) / 20.0,
        ], dtype=np.float32)

    def update_histories(self, success: int, collision: int):
        self.collision_hist.append(collision)
        self.success_hist.append(success)
        self.queue_hist.append(self.queue_len)
        self.delay_hist.append(self.hol_delay)

# =========================
# Realistic Blind Round-Robin Scheduler
# =========================
class CentralizedScheduler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_ue_idx = 0

    def select_actions(self, env):
        num_ues = self.cfg.num_ues
        num_channels = self.cfg.num_channels
        
        access_actions = [0] * num_ues
        channel_actions = [0] * num_ues

        for ch_idx in range(num_channels):
            ue_idx = (self.current_ue_idx + ch_idx) % num_ues
            access_actions[ue_idx] = 1
            channel_actions[ue_idx] = ch_idx
            
        self.current_ue_idx = (self.current_ue_idx + num_channels) % num_ues
        return access_actions, channel_actions

# =========================
# Environment
# =========================
class MACEnvironmentTwoStage:
    def __init__(self, cfg: Config, agent_type: str = "full"):
        self.cfg = cfg
        self.agent_type = agent_type
        self.ues = [UE(i, cfg) for i in range(cfg.num_ues)]

        self.state_dim = 10
        self.access_action_dim = 2
        self.channel_action_dim = cfg.num_channels

        if agent_type == "full":
            self.access_agents = [DQNAgent(self.state_dim, self.access_action_dim, cfg) for _ in range(cfg.num_ues)]
            self.channel_agents = [DQNAgent(self.state_dim, self.channel_action_dim, cfg) for _ in range(cfg.num_ues)]
        elif agent_type == "light":
            self.access_agents = [LightweightDQNAgent(self.state_dim, self.access_action_dim, cfg) for _ in range(cfg.num_ues)]
            self.channel_agents = [LightweightDQNAgent(self.state_dim, self.channel_action_dim, cfg) for _ in range(cfg.num_ues)]

        self.current_slot = 0
        self.reset_stats()

    def reset_stats(self):
        self.total_success = 0
        self.total_collision = 0
        self.total_listen = 0
        self.total_tx = 0

    def reset(self):
        for ue in self.ues:
            ue.reset()
        self.current_slot = 0
        self.reset_stats()
        return [ue.get_state() for ue in self.ues]

    def step(self, access_actions, channel_actions):
        self.current_slot += 1
        for ue in self.ues:
            ue.arrival_step()
            ue.update_delay()

        states = [ue.get_state() for ue in self.ues]

        channel_groups = defaultdict(list)
        actual_channels = [0 for _ in range(self.cfg.num_ues)]

        for i, ue in enumerate(self.ues):
            access_a = access_actions[i]
            if ue.queue_len == 0:
                access_a = 0

            if access_a == 1:
                ch = channel_actions[i] + 1
                actual_channels[i] = ch
                channel_groups[ch].append(i)

        rewards = [0.0 for _ in range(self.cfg.num_ues)]
        results = ["listen" for _ in range(self.cfg.num_ues)]
        info = {"success": 0, "collision": 0, "listen": 0, "tx": 0}

        for i, ue in enumerate(self.ues):
            access_a = access_actions[i]
            if ue.queue_len == 0:
                access_a = 0

            if access_a == 0:
                results[i] = "listen"
                ue.episode_listen += 1
                info["listen"] += 1
                if ue.queue_len == 0:
                    rewards[i] += self.cfg.r_idle_empty
                else:
                    rewards[i] += self.cfg.r_idle_backlog
            else:
                info["tx"] += 1
                ue.episode_tx += 1
                self.total_tx += 1

                selected_channel = actual_channels[i]
                contenders = channel_groups[selected_channel]
                rewards[i] += self.cfg.r_tx_attempt

                if len(contenders) == 1:
                    results[i] = "success"
                    rewards[i] += self.cfg.r_success
                    if ue.queue_len > 0:
                        ue.queue_len -= 1
                    if ue.queue_len == 0:
                        ue.hol_delay = 0
                    ue.episode_success += 1
                    info["success"] += 1
                    self.total_success += 1
                else:
                    results[i] = "collision"
                    rewards[i] += self.cfg.r_collision
                    ue.episode_collision += 1
                    info["collision"] += 1
                    self.total_collision += 1

        for i, ue in enumerate(self.ues):
            rewards[i] += self.cfg.r_queue_penalty * ue.queue_len
            rewards[i] += self.cfg.r_delay_penalty * ue.hol_delay

        for i, ue in enumerate(self.ues):
            access_a = access_actions[i]
            if ue.queue_len == 0 and results[i] == "listen":
                access_a = 0
            ue.last_access_action = access_a
            ue.last_channel_action = actual_channels[i]

            if results[i] == "success":
                ue.last_result = 1
                ue.update_histories(success=1, collision=0)
            elif results[i] == "collision":
                ue.last_result = 2
                ue.update_histories(success=0, collision=1)
            else:
                ue.last_result = 0
                ue.update_histories(success=0, collision=0)
                self.total_listen += 1

        next_states = [ue.get_state() for ue in self.ues]
        done = self.current_slot >= self.cfg.episode_length
        return states, rewards, next_states, done, info, results, actual_channels

# =========================
# Logger & Metrics
# =========================
class ExperimentLogger:
    def __init__(self, name, cfg: Config):
        self.name = name
        self.cfg = cfg
        self.data = []
        self.cumulative_success = 0
        self.cumulative_slots = 0
        
        # 動態創建此 Config 專屬的資料夾
        self.exp_suffix = cfg.get_exp_suffix()
        self.data_dir = os.path.join("result", self.exp_suffix, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def log(self, ep, reward, tx, success, collision, listen, avg_queue, avg_delay, episode_slots):
        self.cumulative_success += success
        self.cumulative_slots += episode_slots
        long_term_throughput = self.cumulative_success / self.cumulative_slots

        self.data.append({
            "episode": ep,
            "reward": reward,
            "tx": tx,
            "success": success,
            "collision": collision,
            "listen": listen,
            "success_ratio": success / max(1, tx),
            "avg_queue": avg_queue,
            "avg_delay": avg_delay,
            "long_term_throughput": long_term_throughput
        })

    def save_csv(self):
        # 檔名也套上動態後綴
        filename = f"{self.name}_{self.exp_suffix}.csv"
        filepath = os.path.join(self.data_dir, filename)
        keys = self.data[0].keys()
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)
        print(f"Saved Data: {filepath}")

def collect_episode_metrics(env):
    total_queue = sum(ue.queue_len for ue in env.ues)
    total_delay = sum(ue.hol_delay for ue in env.ues)
    return {
        "avg_queue": total_queue / len(env.ues),
        "avg_delay": total_delay / len(env.ues),
    }

# =========================
# Training Unified
# =========================
def train_model(cfg: Config, agent_type="full"):
    print(f"\n========== Training {agent_type.upper()} RL ==========")
    set_seed(cfg.seed)

    env = MACEnvironmentTwoStage(cfg, agent_type=agent_type)
    logger_name = "full_dqn" if agent_type == "full" else "light_rl"
    logger = ExperimentLogger(logger_name, cfg)

    episode_rewards = []

    for ep in range(cfg.num_episodes):
        states = env.reset()
        done = False
        ep_reward_sum = 0.0
        ep_success = ep_collision = ep_listen = ep_tx = 0

        while not done:
            access_actions, channel_actions = [], []

            for i in range(cfg.num_ues):
                access_a = env.access_agents[i].select_action(states[i], training=True)
                channel_a = env.channel_agents[i].select_action(states[i], training=True)
                access_actions.append(access_a)
                channel_actions.append(channel_a)

            curr_states, rewards, next_states, done, info, results, actual_channels = env.step(
                access_actions, channel_actions
            )

            for i in range(cfg.num_ues):
                if agent_type == "full":
                    env.access_agents[i].store_transition(curr_states[i], access_actions[i], rewards[i], next_states[i], float(done))
                    env.access_agents[i].train_step()
                elif agent_type == "light":
                    env.access_agents[i].train_step_online(curr_states[i], access_actions[i], rewards[i], next_states[i], float(done))

                if access_actions[i] == 1:
                    if agent_type == "full":
                        env.channel_agents[i].store_transition(curr_states[i], channel_actions[i], rewards[i], next_states[i], float(done))
                        env.channel_agents[i].train_step()
                    elif agent_type == "light":
                        env.channel_agents[i].train_step_online(curr_states[i], channel_actions[i], rewards[i], next_states[i], float(done))

            states = next_states
            ep_reward_sum += sum(rewards)
            ep_success += info["success"]
            ep_collision += info["collision"]
            ep_listen += info["listen"]
            ep_tx += info["tx"]

        episode_rewards.append(ep_reward_sum)

        extra = collect_episode_metrics(env)
        logger.log(ep, ep_reward_sum, ep_tx, ep_success, ep_collision, ep_listen, extra["avg_queue"], extra["avg_delay"], cfg.episode_length)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:4d} | Reward: {np.mean(episode_rewards[-10:]):.2f} | Success: {ep_success} | Collision: {ep_collision}")

    logger.save_csv()
    return env

# =========================
# Evaluate Baseline
# =========================
def run_round_robin(cfg: Config):
    print(f"\n========== Running Round-Robin Baseline ==========")
    set_seed(cfg.seed)

    env = MACEnvironmentTwoStage(cfg, agent_type="full")
    scheduler = CentralizedScheduler(cfg)
    logger = ExperimentLogger("round_robin", cfg)

    episode_rewards = []

    for ep in range(cfg.num_episodes):
        env.reset()
        scheduler.current_ue_idx = 0 
        done = False
        
        ep_reward_sum = 0.0
        ep_success = ep_collision = ep_listen = ep_tx = 0

        while not done:
            access_actions, channel_actions = scheduler.select_actions(env)
            _, rewards, _, done, info, _, _ = env.step(access_actions, channel_actions)

            ep_reward_sum += sum(rewards)
            ep_success += info["success"]
            ep_collision += info["collision"]
            ep_listen += info["listen"]
            ep_tx += info["tx"]

        episode_rewards.append(ep_reward_sum)
        extra = collect_episode_metrics(env)
        logger.log(ep, ep_reward_sum, ep_tx, ep_success, ep_collision, ep_listen, extra["avg_queue"], extra["avg_delay"], cfg.episode_length)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:4d} | Reward: {np.mean(episode_rewards[-10:]):.2f} | Success: {ep_success} | Collision: {ep_collision}")

    logger.save_csv()

# =========================
# Plotting
# =========================
def plot_comparison(cfg: Config):
    print("\n========== Generating Plots ==========")
    exp_suffix = cfg.get_exp_suffix()
    
    # 讀取路徑與儲存路徑動態化
    data_dir = os.path.join("result", exp_suffix, "data")
    pic_dir = os.path.join("result", exp_suffix, "pic")
    os.makedirs(pic_dir, exist_ok=True)

    def save_and_show_plot(metric_name):
        filename = f"{metric_name}_{exp_suffix}.png"
        filepath = os.path.join(pic_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved Plot: {filepath}")
        plt.show()

    try:
        full = pd.read_csv(os.path.join(data_dir, f"full_dqn_{exp_suffix}.csv"))
        light = pd.read_csv(os.path.join(data_dir, f"light_rl_{exp_suffix}.csv"))
        rr = pd.read_csv(os.path.join(data_dir, f"round_robin_{exp_suffix}.csv"))
    except FileNotFoundError:
        print(f"CSV not found in {data_dir}, run training first.")
        return

    # 1. Success Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(full["episode"], full["success"], label="Full DQN", linewidth=2, zorder=2)
    plt.plot(light["episode"], light["success"], label="Lightweight RL", linewidth=2, linestyle="--", zorder=2)
    plt.plot(rr["episode"], rr["success"], label="Blind Round-Robin", linewidth=3, linestyle="-.", color='green', zorder=3)
    plt.xlabel("Episode")
    plt.ylabel("Success Transmissions")
    plt.title(f"Success Comparison ({exp_suffix})")
    plt.legend()
    plt.grid(True)
    save_and_show_plot("success")

    # 2. Collision Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(full["episode"], full["collision"], label="Full DQN", linewidth=2, zorder=2)
    plt.plot(light["episode"], light["collision"], label="Lightweight RL", linewidth=2, linestyle="--", zorder=2)
    plt.plot(rr["episode"], rr["collision"], label="Blind Round-Robin", linewidth=3, linestyle="-.", color='green', zorder=3)
    plt.xlabel("Episode")
    plt.ylabel("Collisions")
    plt.title(f"Collision Comparison ({exp_suffix})")
    plt.ylim(bottom=-5) 
    plt.legend()
    plt.grid(True)
    save_and_show_plot("collision")
    
    # 3. Queue Length Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(full["episode"], full["avg_queue"], label="Full DQN", linewidth=2, zorder=2)
    plt.plot(light["episode"], light["avg_queue"], label="Lightweight RL", linewidth=2, linestyle="--", zorder=2)
    plt.plot(rr["episode"], rr["avg_queue"], label="Blind Round-Robin", linewidth=3, linestyle="-.", color='green', zorder=3)
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.title(f"Queue Management ({exp_suffix})")
    plt.ylim(bottom=-0.5)
    plt.legend()
    plt.grid(True)
    save_and_show_plot("queue")

    # 4. Long Term Throughput Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(full["episode"], full["long_term_throughput"], label="Full DQN", linewidth=2, zorder=2)
    plt.plot(light["episode"], light["long_term_throughput"], label="Lightweight RL", linewidth=2, linestyle="--", zorder=2)
    plt.plot(rr["episode"], rr["long_term_throughput"], label="Blind Round-Robin", linewidth=3, linestyle="-.", color='green', zorder=3)
    plt.xlabel("Episode")
    plt.ylabel("Long Term Throughput (Success/Slot)")
    plt.title(f"Long Term Throughput ({exp_suffix})")
    plt.legend(loc='lower right')
    plt.grid(True)
    save_and_show_plot("throughput")

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # 建立唯一的 Config 物件，保證大家參數一致
    # 你可以在這裡直接改參數，例如 cfg = Config(num_channels=2, num_ues=8, arrival_lambda=2.5)
    cfg = Config()
    
    # 1. 訓練 Full DQN
    train_model(cfg, agent_type="full")

    # 2. 訓練 Lightweight RL
    train_model(cfg, agent_type="light")
    
    # 3. 跑 Blind Round-Robin 基準
    run_round_robin(cfg)

    # 4. 畫圖表比較
    plot_comparison(cfg)