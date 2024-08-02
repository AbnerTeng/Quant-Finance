# %%
import gym
import numpy as np
from tqdm import tqdm
# %%
env = gym.make('FrozenLake-v1', is_slippery=False)
state_space = env.observation_space.n
action_space = env.action_space.n
q_table = np.zeros((state_space, action_space))

lr = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 10
# %%
def q_learning(env, episodes, lr, gamma, epsilon):
    for epi in tqdm(range(episodes)):
        state = env.reset()[0]
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        best_next_action = np.argmax(q_table[next_state, :])
        q_table[state, action] = q_table[state, action] + lr * (
            reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
        )
        state = next_state

def test_agent(env, episodes=5):
    for epi in tqdm(range(episodes)):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(q_table[state, :])
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {epi + 1}: Total reward = {total_reward}")

# %%
if __name__ == "__main__":
    q_learning(env, episodes, lr, gamma, epsilon)
    test_agent(env)
# %%
class CustomEnv(gym.Env):
    def __init__(self, data):
        super(CustomEnv, self).__init__()
        self.data = data
        self.action_space = gym.spaces.Discrete(N)