import torch
import torch.nn as nn
import random

# 强化学习环境定义
class RecommendationEnv:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        self.current_state = random.choice(self.data)
        return self.current_state

    def step(self, action):
        reward = random.uniform(0, 1) if action == self.current_state else -0.1
        next_state = random.choice(self.data)
        return next_state, reward, False

# 策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化环境和策略网络
env = RecommendationEnv(processed_data)
policy_net = PolicyNetwork(input_dim=768 + 512 + 512, output_dim=len(processed_data))
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

# 强化学习训练
for episode in range(100):
    state = env.reset()
    total_reward = 0
    for t in range(10):
        action = torch.argmax(policy_net(torch.tensor(state).float())).item()
        next_state, reward, done = env.step(action)
        total_reward += reward

        # 更新策略网络
        loss = -reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break
        state = next_state

    print(f"Episode {episode + 1}, Total Reward: {total_reward:.4f}")

# 保存强化学习策略网络
torch.save(policy_net.state_dict(), "models/reinforcement_model.pth")
print("Reinforcement learning training completed and saved to 'models/reinforcement_model.pth'")
