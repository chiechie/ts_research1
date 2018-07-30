# -*- coding: utf-8 -*-
import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
from tensorboard_helper import variable_summaries
# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
REPLAY_SIZE = 20000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch

class DQN(object):
	# DQN Agent
	def __init__(self, data_dictionary):
		#pdb.set_trace();
		# init experience replay
		self.replay_buffer = deque()
		# init some parameters
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = data_dictionary["input"]
		self.action_dim = data_dictionary["action"]

		self.create_Q_network(data_dictionary)
		self.create_training_method()

		# Init session
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())

		# loading networks
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
				print "Could not find old network weights"

		global summary_writer
		summary_writer = tf.summary.FileWriter('logs', graph=self.session.graph)

	def create_Q_network(self, data_dictionary):
		# network weights
		W1 = self.weight_variable([self.state_dim, data_dictionary["hidden_layer_1_size"]])
		variable_summaries(W1, "layer1/weights")
		b1 = self.bias_variable([data_dictionary["hidden_layer_1_size"]])
		variable_summaries(b1, "layer1/bias")
		W2 = self.weight_variable([data_dictionary["hidden_layer_1_size"],self.action_dim])
		variable_summaries(W2, "layer2/weights")
		b2 = self.bias_variable([self.action_dim])
		variable_summaries(b2, "layer2/bias")
		#tf.scalar_summary("second_layer_bias_scaler", b2)
		self.b2 = b2
		# input layer
		self.state_input = tf.placeholder("float", [None,self.state_dim])
		# hidden layers
		h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
		# Q Value layer
		self.Q_value = tf.matmul(h_layer, W2) + b2

	def create_training_method(self):
		self.action_input = tf.placeholder("float", [None, self.action_dim]) # one hot presentation
		self.y_input = tf.placeholder("float", [None])
		Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input),reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		tf.summary.scalar("loss", self.cost)
		global merged_summary_op
		merged_summary_op = tf.summary.merge_all()
		self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

	def perceive(self, state, action, reward, next_state, done):
		self.time_step += 1
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()

		if len(self.replay_buffer) > 2000:
			self.train_Q_network()

	def train_Q_network(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		next_state_batch = [data[3] for data in minibatch]

		# Step 2: calculate y
		y_batch = []
		#pdb.set_trace();
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
		for i in range(0, BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input: y_batch,
			self.action_input: action_batch,
			self.state_input: state_batch
			})
		summary_str = self.session.run(merged_summary_op,feed_dict={
				self.y_input: y_batch,
				self.action_input: action_batch,
				self.state_input: state_batch
				})
		summary_writer.add_summary(summary_str, self.time_step)
		#pdb.set_trace()

		# save network every 1000 iteration
		if self.time_step % 1000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.time_step)

	def egreedy_action(self, state):
		Q_value = self.Q_value.eval(feed_dict={
			self.state_input: [state]
			})[0]
		#print(self.time_step)
		#print(self.epsilon)
		if self.time_step > 200000:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/1000000
		if random.random() <= self.epsilon:
			return random.randint(0, self.action_dim - 1)
		else:
			return np.argmax(Q_value)

	def action(self, state):
		return np.argmax(self.Q_value.eval(feed_dict = {
			self.state_input: [state]
			})[0])

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

