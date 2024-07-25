# main.py
import numpy as np
import tensorflow as tf
from environment import SimpleEnvironment
from policy_network import PolicyNetwork
from helpers import get_action, compute_returns, train_step

def main():
    env = SimpleEnvironment()
    state_size = 1  # because the state is a single number
    action_size = 2  # two possible actions: increment or decrement
    policy = PolicyNetwork(state_size, action_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Training
    episodes = 500
    gamma = 0.99

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []

        done = False
        while not done:
            action = get_action(policy, state, action_size)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards, gamma)
        train_step(policy, optimizer, states, actions, returns, action_size)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

    # Testing the trained policy
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = get_action(policy, state, action_size)
        state, reward, done = env.step(action)
        total_reward += reward

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
