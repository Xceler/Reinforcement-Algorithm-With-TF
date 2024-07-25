import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Define the simple custom environment
class SimpleEnvironment:
    def __init__(self):
        self.state = 0
        self.end_state = 5

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state -= 1

        if self.state == self.end_state:
            return self.state, 1, True  # reward of 1 for reaching the end state
        elif self.state < 0 or self.state > self.end_state:
            return self.state, -1, True  # reward of -1 for going out of bounds
        else:
            return self.state, 0, False  # no reward otherwise

    def render(self):
        print(f"State: {self.state}")

# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(24, activation='relu')
        self.dense2 = layers.Dense(24, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Helper functions for the REINFORCE algorithm
def get_action(policy, state):
    state = np.expand_dims(state, axis=0)  # Ensure state is a 2D tensor
    state = np.expand_dims(state, axis=-1) # Add feature dimension
    probabilities = policy(tf.convert_to_tensor(state, dtype=tf.float32))
    return np.random.choice(action_size, p=np.squeeze(probabilities))

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return returns

def train_step(policy, optimizer, states, actions, returns):
    states = np.array(states).reshape(-1, 1)  # Reshape states to be 2D
    with tf.GradientTape() as tape:
        logits = policy(tf.convert_to_tensor(states, dtype=tf.float32))
        action_masks = tf.one_hot(actions, action_size)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
        loss = -tf.reduce_mean(log_probs * returns)
    
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

# Main script
if __name__ == "__main__":
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
            action = get_action(policy, state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards, gamma)
        train_step(policy, optimizer, states, actions, returns)

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

    # Testing the trained policy
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = get_action(policy, state)
        state, reward, done = env.step(action)
        total_reward += reward

    print(f"Total Reward: {total_reward}")

