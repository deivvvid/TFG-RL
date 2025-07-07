import torch
import torch.nn as nn
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.model(x)

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

env = CartPoleEnv(render_mode="human")
env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

q_network = QNetwork(obs_space, action_space)
q_network.load_state_dict(torch.load("best_dqn_cartpole.pth"))
q_network.eval()

state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    action = q_values.argmax().item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    state = next_state
    total_reward += reward
    done = terminated or truncated

print(f"\n✅ Test episode finished with total reward: {total_reward}")
env.close()
