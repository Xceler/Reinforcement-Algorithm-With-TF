from environment import SimpleEnvironment 
from policy_network import PolicyNetwork 
from helpers import get_action, compute_returns, train_step 
import tensorflow as tf 


if __name__ == '__main__':
    env = SimpleEnvironment()
    state_size = 1 
    action_size= 2 
    policy = PolicyNetwork(state_size, action_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

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

        returns = compute_returns(rewards, gamma) 
        train_step(policy, optimizer, state, actions, returns)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")
        
    state = env.reset() 
    done = False 
    total_reward = 0 
    while not done:
        env.render()
        action = get_action(policy, state)
        state, reward, done = env.step(action)
        total_reward += reward 
    
    print(f"TOtal Reward: {total_reward}")