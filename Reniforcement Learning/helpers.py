import numpy as np 
import tensorflow as tf 

def get_action(policy, state, action_size):
    state = np.expand_dims(state, axis = 0)
    state = np.expand_dims(state, axis = -1)
    probabilities = policy(tf.convert_to_tensor(state, dtype = tf.float32))
    return np.random.choice(action_size, p = np.squeeze(probabilities))

def compute_returns(rewards, gamma = 0.99):
    returns = [] 
    R = 0 
    for reward in reversed(rewards):
        R = reward + gamma * R 
        returns.insert(0, R)
    return returns 

def train_step(policy, optimizer, states, actions, returns):
    states = np.array(states).reshape(-1, 1)
    with tf.GradientTape() as tape:
        logits = policy(tf.convert_to_tensor(states, dtype = tf.float32))
        action_masks = tf.one_hot(actions, action_size)
        log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis = 1)
        loss = -tf.reduce_mean(log_probs * returns)
    
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads,policy.trainable_variables))
