"""
Title: Deep Reinforcement Learning for Continuous-Time 
       Self-Triggered Control with Experimental Evaluation
Author: Ran,Wang
Date created: 2021/11/20
Last modified: 2022/01/21
Description: Implementing Deep Reinforcement Learning for
             Continuous-time Self-triggered Control (DDPG4STC)
"""
import gym,myenv,os
import tensorflow as tf
from utils import ActorModel,CriticModel,mkdir,process_bar,ReplayBuffer,STCActionNoise
import numpy as np
import pickle
from typing import TypeVar

SelfDDPG4STC = TypeVar("SelfDDPG4STC", bound="DDPG4STC")

class DDPG4STC():
    def __init__(self,tau0:float = 0.1, # init tau
                problem:str = "RotaryPend-v0", # gym env 
                name:str = 'default', # name of the agent
                learning_rate:list=[0.001,0.0001,0.0005] # lr for Q,u,tau
                ):

        self.problem = problem
        self.env = gym.make(self.problem)
        self.name = problem.replace('-','_') + '_' + name

        self.actor = ActorModel(init_tau = tau0)    # pi(x|omega^pi)
        self.critic = CriticModel()  # Q(x,a|omega^Q)

        self.target_actor = ActorModel()   #pi'(x|omega^pi')
        self.target_critic = CriticModel() #Q'(x,a|omega^Q')
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        '''learning rate'''
        Q_lr = learning_rate[0]
        u_lr = learning_rate[1]
        tau_lr = learning_rate[2]

        '''Adam'''
        self.critic_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.u_optimizer = tf.keras.optimizers.Adam(u_lr)
        self.tau_optimizer = tf.keras.optimizers.Adam(tau_lr)
        
        # Step discount rate of STC is e^(-alpha * dt)
        self.alpha = 0.2
        # communication cost constant
        self.beta = 0.4
        # update rate for target network
        self.sigma = 0.01
        # replay buffer
        self.buffer = ReplayBuffer(buffer_capacity = 500000, batch_size = 128)
        # STC action noise
        self.noise = STCActionNoise()

    def load(self,version:str='v0'):
        '''load check point'''
        path = './data/{}_{}/'.format(self.name,version) 
        self.buffer = pickle.load(open(path + 'buffer.replay','rb'))
        self.actor.load_weights(path + "actor.h5")
        self.critic.load_weights(path + "critic.h5")
        self.target_actor.load_weights(path + "target_actor.h5")
        self.target_critic.load_weights(path + "target_critic.h5")
    
    def save(self,version:str='v0'):
        '''save check point'''
        path = './data/{}_{}/'.format(self.name,version) 
        mkdir(path)
        pickle.dump(self.buffer,open(path + 'buffer.replay','wb'))
        self.actor.save_weights(path + "actor.h5")
        self.critic.save_weights(path + "critic.h5")
        self.target_actor.save_weights(path + "target_actor.h5")
        self.target_critic.save_weights(path + "target_critic.h5")

    @tf.function
    def update_target(self, target_weights, weights, sigma):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * sigma + a * (1 - sigma))

    @tf.function 
    def update_u(self,state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape() as tape:
            u_target, tau_target = self.target_actor(next_state_batch, training=True)
            gamma = tf.expand_dims(tf.exp(-self.alpha * tau_target),1)
            y_target = gamma * self.target_critic(
                [next_state_batch, u_target, tau_target], training=True
            )
            y = reward_batch + y_target
            u_batch, tau_batch= tf.split(action_batch,2,axis=1)
            critic_value = self.critic([state_batch, u_batch, tau_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            u, tau = self.actor(state_batch, training=True)
            Q = self.critic([state_batch, u,tau], training=True)
            
            # use -Q to maximize -Q (minimize Q)
            u_loss = -tf.math.reduce_mean(Q)
            

        u_grad = tape.gradient(u_loss, self.actor.vars_u)
        self.u_optimizer.apply_gradients(
            zip(u_grad, self.actor.vars_u)
        )

    @tf.function 
    def update_tau(self,state_batch, action_batch, reward_batch, next_state_batch,Vx_prime):
        with tf.GradientTape() as tape:
            u_target, tau_target = self.target_actor(next_state_batch, training=True)
            gamma = tf.expand_dims(tf.exp(-self.alpha * tau_target),1)
            y_target = gamma * self.target_critic(
                [next_state_batch, u_target, tau_target], training=True
            )
            y = reward_batch + y_target
            u_batch, tau_batch= tf.split(action_batch,2,axis=1)
            critic_value = self.critic([state_batch, u_batch, tau_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            u, tau = self.actor(state_batch, training=True)
            Q =  self.critic([state_batch, u,tau], training=True)
            
            tau_theta_term =  tf.exp(-self.alpha * tau) * Vx_prime

            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            tau_loss = -tf.math.reduce_mean(Q + tau_theta_term)

        tau_grad = tape.gradient(tau_loss, self.actor.vars_tau)
        self.tau_optimizer.apply_gradients(
             zip(tau_grad, self.actor.vars_tau)
        )

    def learn(self,ep,tau_ep=1, sum_ep= 100):
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.get_minibatch()
        #only update tau at 1st step of every N_u + N_tau steps
        if(ep % sum_ep >= tau_ep):
            self.update_u(state_batch, action_batch, reward_batch, next_state_batch)
        else:
            u_prime,tau_prime = self.actor(next_state_batch)
            Vx_prime = self.critic([next_state_batch, u_prime,tau_prime])
            self.update_tau(state_batch, action_batch, reward_batch, next_state_batch,Vx_prime)
          
    def policy(self,state, IsExploration=True):
        actions = tf.squeeze(self.actor(state))
        
        # Adding noise to action
        if IsExploration:
            self.noise.dt=float(actions[1])
            actions = actions.numpy() +  self.noise()

        # We make sure action is within bounds add triggerting time h bound
        legal_actions = np.clip(actions, [-8, 0.01], [8, 2.00])

        return np.squeeze(legal_actions)

    def train_start(self,Ne=10, Ntau=0, Nu=99):
        Nsum = Ntau+Nu

        # To store reward history of each episode
        ep_cost_list, ep_comcost_list = [],[]

        for ep in range(Ne):          
            prev_state = self.env.reset()
            episodic_reward = 0
            episodic_comcost= 0

            t = 0
            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state)
                com_cost = self.beta
                stage_reward = - com_cost  
                #stage reward for STC (communication cost is included here)
                last_t = t
                tau = float(action[1]) #triggering time interval
                # Recieve state and reward from environment.
                while(t - last_t <= tau):
                    state, reward, done, info = self.env.step([action[0]])
                    stage_reward += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))
                    t += self.env.dt
                    if done:
                        break
                    #self.env.render()
                
                if done:
                    tmp = np.exp(- self.alpha * t) * (1/(1-np.exp(- self.alpha * tau)))
                    episodic_reward +=  stage_reward * tmp
                    episodic_comcost+= com_cost * tmp
                    break

                self.buffer.record((prev_state, action, stage_reward, state))
                stage_reward *= np.exp(- self.alpha * last_t)
                com_cost *= np.exp(- self.alpha * last_t)
                episodic_reward += stage_reward 
                episodic_comcost+= com_cost

                self.learn(ep,tau_ep= Ntau, sum_ep= Nsum) #tau_ep=0 means only train actor_u
                self.update_target(self.target_actor.variables, self.actor.variables, self.sigma)
                self.update_target(self.target_critic.variables, self.critic.variables, self.sigma)
 
                prev_state = state

            self.noise.reset()
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            '''negtive episodic reward = episodic cost'''
            ep_cost_list.append(-episodic_reward)
            ep_comcost_list.append(episodic_comcost)

            # Mean of last 100 episodes
            avg_cost = np.mean(ep_cost_list[-100:])
            avg_comcost = np.mean(ep_comcost_list[-100:])
     
            process_bar(ep/Ne,total_length=50,end_str='ep={},cost={}[{}]    '.format(ep,round(avg_cost,3),round(avg_comcost,3)))
         


    def algorithm_ptc(self,Ne:int=10000):
        self.train_start(Ne=Ne, Ntau=0, Nu=99)
        self.save(version = 'ptc')

    def algorithm_stc(self,Ne:int=10000):
        self.load(version = 'ptc')
        self.train_start(Ne=Ne, Ntau=1, Nu=499)
        self.save(version = 'stc')

if __name__ == '__main__':   
    pass