import tensorflow as tf 
from tensorflow.keras import layers 


class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(24, activation = 'relu')
        self.dense2 = layers.Dense(24, activation = 'relu')
        self.output_layer = layers.Dense(action_size, activation = 'softmax')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
    
    