import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, action_dim=1, obs_dim=5, buffer_capacity=50000, batch_size=64):
        self.buffer_capacity = buffer_capacity #N_R
        self.batch_size = batch_size           #N_m
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, obs_dim))      # [theta,theta_dot,cos(phi),sin(phi),phi_dot]
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim+1))     # [u, tau]
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))     # [stage cost]
        self.next_state_buffer = np.zeros((self.buffer_capacity, obs_dim)) # [theta,theta_dot,cos(phi),sin(phi),phi_dot]

    # (s,a,r,s') obervation tuples
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def get_minibatch(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        return state_batch, action_batch, reward_batch, next_state_batch

    def reset(self):
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, 5))      # [theta,theta_dot,cos(phi),sin(phi),phi_dot]
        self.action_buffer = np.zeros((self.buffer_capacity, 2))     # [u, tau]
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))     # [stage cost]
        self.next_state_buffer = np.zeros((self.buffer_capacity, 5)) # [theta,theta_dot,cos(phi),sin(phi),phi_dot]