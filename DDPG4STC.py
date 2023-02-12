"""
Title: Deep reinforcement learning for self-triggered control applied to underactuated systems
Author: Ran,Wang
Date created: 2021/11/20
Last modified: 2022/01/21
Description: Implementing Deep Reinforcement Learning for Continuous-time Self-triggered Control
             on a rotary inverted pendulum.
"""
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense,Concatenate,Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pickle
import myenv

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

class ActorModel(Model):
  def __init__(self,init_tau = 0.16):
    ''' init_tau is from [0.04, 0.50] '''
    super().__init__()
    init_tau = np.clip(init_tau, 0.01, 0.50)
    init_tau = np.arctanh(init_tau - 1.000001)


    u_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    tau_init = tf.constant_initializer(0)
    tau_init_bias = tf.constant_initializer(init_tau)

    self.d11 = Dense(128, activation="relu",name='u11')
    self.d12 = Dense(128, activation="relu",name='u12')
    self.d13 = Dense(1, activation="tanh",name='u',kernel_initializer = u_init)

    self.d21 = Dense(128, activation="relu",name='tau11')
    self.d22 = Dense(128, activation="relu",name='tau12')
    self.d23 = Dense(1, activation="tanh",name='tau',kernel_initializer=tau_init,bias_initializer=tau_init_bias)
    self.__call__(Input(5))
    self.vars_u    = self.trainable_variables[0:6]
    self.vars_tau =  self.trainable_variables[6:12]

  def call(self, inputs):
    # Input for pendulum is  [cos(theta) sin(theta) dot_theta]
    x1 = self.d11(inputs)
    x1 = self.d12(x1)
    u = self.d13(x1)   
    u = 8 * u                     # torque u [N.m] (-2,2)
    
    x2 = self.d21(inputs)
    x2 = self.d22(x2)
    tau = self.d23(x2) 
    tau = tau + K.constant(1.0)  # triggering time tau [sec] (0.02,2.02)
    return u,tau

class CriticModel(Model):
  def __init__(self):
    super().__init__()

    self.concat_u = Concatenate()
    self.concat_tau = Concatenate()
    self.concat_Q = Concatenate()
    
    self.du1 = Dense(256, activation="relu")
    self.du2 = Dense(256, activation="relu")
    self.du3 = Dense(1, activation="linear")
    self.dtau1 = Dense(256, activation="relu")
    self.dtau2 = Dense(256, activation="relu")
    self.dtau3 = Dense(1, activation="linear")

    self.dQ = Dense(1, activation="linear")

    self.__call__([Input(5),Input(1),Input(1)])

  def call(self, inputs):
    inputs_u_x = self.concat_u([inputs[0],inputs[1]])
    inputs_tau_x = self.concat_tau([inputs[0],inputs[2]])
    
    Q_u = self.du1(inputs_u_x)
    Q_u = self.du2(Q_u)
    Q_u = self.du3(Q_u)
    
    Q_tau = self.dtau1(inputs_tau_x)
    Q_tau = self.dtau2(Q_tau)
    Q_tau = self.dtau3(Q_tau)

    Q = self.concat_Q([Q_u,Q_tau])

    out =  self.dQ(Q) #Qπ
    return out

class DDPG4STC():
    def __init__(self,tau0 = 0.1,problem = "RotaryPend-v0", path = './'):
        self.problem = problem
        self.env = gym.make(self.problem)

        self.path = path # directory for save or load the variables and replay buffer
        self.mkdir()     # if path does not exist, make dir. 

        self.actor = ActorModel(init_tau = tau0)    # pi(x|omega^pi)
        self.critic = CriticModel()  # Q(x,a|omega^Q)

        self.target_actor = ActorModel()   #pi'(x|omega^pi')
        self.target_critic = CriticModel() #Q'(x,a|omega^Q')
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        '''learning rate'''
        Q_lr = 0.001
        u_lr = 0.0001
        tau_lr = 0.00005

        '''Adam'''
        self.critic_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.u_optimizer = tf.keras.optimizers.Adam(u_lr)
        self.tau_optimizer = tf.keras.optimizers.Adam(tau_lr)
        
        # Step discount rate of STC is e^(-alpha * dt)
        self.alpha = 0.2
        # communication cost constant
        self.beta = 0.4
        # update ragte for target network
        self.sigma = 0.01
        # replay buffer
        self.buffer = self.ReplayBuffer(buffer_capacity = 500000, batch_size = 128)
        # STC action noise
        self.noise = self.STCActionNoise()

    def process_bar(self, percent, start_str='', end_str='', total_length=30):
        bar = ''.join(['■'] * int(percent * total_length)) + ''
        bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
        print(bar, end='', flush=True)

    def mkdir(self):
        path = self.path
        path = path.strip()
        path = path.rstrip("\\")

        isExists=os.path.exists(path)
    
        if not isExists:
            os.makedirs(path) 
            print('path:' + path + ' done')
            return True
        else:
            print('path:' + path)
            return False

    def load(self):
        '''load check point'''
        path = self.path
        self.buffer = pickle.load(open(path + 'buffer_rotarypend.replay','rb'))
        self.actor.load_weights(path + "rotarypend_actor.h5")
        self.critic.load_weights(path + "rotarypend_critic.h5")
        self.target_actor.load_weights(path + "rotarypend_target_actor.h5")
        self.target_critic.load_weights(path + "rotarypend_target_critic.h5")
    
    def save(self):
        '''save check point'''
        path = self.path
        pickle.dump(self.buffer,open(path + 'buffer_rotarypend.replay','wb'))
        self.actor.save_weights(path + "rotarypend_actor.h5")
        self.critic.save_weights(path + "rotarypend_critic.h5")
        self.target_actor.save_weights(path + "rotarypend_target_actor.h5")
        self.target_critic.save_weights(path + "rotarypend_target_critic.h5")

    class STCActionNoise:
        '''
            the noise for self-triggered control input including two parts
            Input u             : Ornstein-Uhlenbeck_process
            Triggering time tau : White noise
        '''
        def __init__(self, mean = [0,0], std_deviation = [0.1,0.05], theta=0.15, dt=0.01):
            # Noise for u
            self.theta = theta
            self.miu_u = mean[0]
            self.sigma_u = std_deviation[0]
            self.dt = dt
            self.reset()
            # Noise for tau
            self.miu_tau = mean[1]
            self.sigma_tau = std_deviation[1]

        def __call__(self):
            # Ornstein-Uhlenbeck_process: https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
            # dxt = theta * (miu - x_t)dt + sigma dWt
            x = (
                self.x_prev
                + self.theta * (self.miu_u - self.x_prev) * self.dt
                + self.sigma_u * np.sqrt(self.dt) * np.random.normal()
            )
            self.x_prev = x

            # Gaussian noise : https://en.wikipedia.org/wiki/Gaussian_noise
            return x, np.random.normal(self.miu_tau,self.sigma_tau)

        def reset(self):
            self.x_prev = self.miu_u

    class ReplayBuffer:
        def __init__(self, buffer_capacity=50000, batch_size=64):
            self.buffer_capacity = buffer_capacity #N_R
            self.batch_size = batch_size           #N_m
            self.buffer_counter = 0

            self.state_buffer = np.zeros((self.buffer_capacity, 5))      # [theta,theta_dot,cos(phi),sin(phi),phi_dot]
            self.action_buffer = np.zeros((self.buffer_capacity, 2))     # [u, tau]
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))     # [stage cost]
            self.next_state_buffer = np.zeros((self.buffer_capacity, 5)) # [theta,theta_dot,cos(phi),sin(phi),phi_dot]

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

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

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
          
    def policy(self,state):
        sampled_actions = tf.squeeze(self.actor(state))
        
        self.noise.dt=float(sampled_actions[1])
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() +  self.noise()

        # We make sure action is within bounds add triggerting time h bound
        legal_action = np.clip(sampled_actions, [-8, 0.01], [8, 2.00])

        return np.squeeze(legal_action)

    def train_start(self,Ne=10, IsNew=True, Ntau=0, Nu=99, IsPrint = False):
        Na = Ntau+Nu
        if not IsNew:
            self.load()
        self.env.set_noise(True)

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        # Takes about 2.5 hours to train
        for ep in range(Ne):

            IsUp= bool(np.random.randint(0,2))
            prev_state = self.env.reset(IsUp=IsUp)
            tf_state0 = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            episodic_reward = 0
            episodic_comcost= 0

            t = 0
            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state)
                com_cost = self.beta
                stage_cost = - self.beta   #stage cost for STC (communication cost is included here)
                last_t = t

                # Recieve state and reward from environment.
                while(t - last_t <= float(action[1])):
                    state, reward, done, info = self.env.step([action[0]])
                    stage_cost += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))
                    t += self.env.dt
                    if done:
                        break
                    #self.env.render()
                
                self.buffer.record((prev_state, action, stage_cost, state))
                stage_cost *= np.exp(- self.alpha * last_t)
                com_cost *= np.exp(- self.alpha * last_t)
                episodic_reward += stage_cost 
                episodic_comcost+= com_cost

                self.learn(ep,tau_ep= Ntau, sum_ep= Na) #tau_ep=0 means only train actor_u
                self.update_target(self.target_actor.variables, self.actor.variables, self.sigma)
                self.update_target(self.target_critic.variables, self.critic.variables, self.sigma)

                # End this episode when `done` is True
                if done:
                    break
                prev_state = state
            self.noise.reset()
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action0 = self.actor(tf_state0)
            action1 = self.actor(tf_prev_state)
            eva_epi_cost = self.critic((tf_state0,action0[0],action0[1])) - np.exp(-self.alpha * 10) * self.critic((tf_prev_state,action1[0],action1[1]))
            eva_epi_cost = -float(eva_epi_cost)
            ep_reward_list.append(-episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            if(IsPrint):
                print("Episode * {} * Avg Reward is ==> {}[{}][{}][{}]".format(ep, avg_reward,-episodic_reward,episodic_comcost,eva_epi_cost))
            else:
                self.process_bar(ep/Ne,total_length=50,end_str='ep={},cost={}[{}]'.format(ep,round(avg_reward,3),round(t,2)))
            avg_reward_list.append(avg_reward)


        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.savefig(self.path + 'train.png')
        plt.close()
        # Save the weights

    def algorithm1(self,Ne=10000):
        self.train_start(Ne=Ne, IsNew=True, Ntau=0, Nu=99)
        self.save()

    def algorithm2(self,Ne=10000):
        self.train_start(Ne=Ne, IsNew=False, Ntau=1, Nu=499)
        self.save()

    def simulation(self,IsPlot=True,x0=[0,0,-np.pi,0],IsNoise=True,IsZeroInput=False):
        self.load()
        self.noise = self.STCActionNoise(std_deviation=[0,0])
        prev_state = self.env.reset(IsUp=False,x0=x0)
        self.env.set_noise(IsNoise)        
        episodic_reward = 0
        theta,theta_dot, Cphi, Sphi, phi_dot = [x0[0]],[x0[1]],[np.cos(x0[2])],[np.sin(x0[2])],[x0[3]]
        u_list,tau_list,t_list  = [0],[0],[0]
        cost_list,stagecost_list,trigger= [0],[0],[1]
        t = 0
        while True:
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state)
            if(IsPlot):
                print(action)
            if(IsZeroInput):
                action[0]=0
            stage_cost = - self.beta   #reward for STC (including communication cost)
            last_t = t
            # Recieve state and reward from environment.
            while(t - last_t <= float(action[1])):
                self.env.render()
                state, reward, done, info = self.env.step([action[0]])
                
                stage_cost += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))

                t += self.env.dt
                t_list.append(t)
                u_list.append(action[0])
                tau_list.append(action[1])
                theta.append(state[0])
                theta_dot.append(state[1])
                Cphi.append(state[2])
                Sphi.append(state[3])
                phi_dot.append(state[4])

                stagecost_list.append(-stage_cost)
                cost_list.append(-episodic_reward)
                trigger.append(0)
                if(done):
                    break
            trigger[-1] = 1
            trigger[-2] = 1

            
            episodic_reward += stage_cost * np.exp(- self.alpha * last_t)

            # End this episode when `done` is True
            if done:
                break
            prev_state = state
        

        if(IsPlot == True):
            #plot with latex style
            #plt.rc('text', usetex=True) #use latex
            fig, axs = plt.subplots(2, 1, figsize=(8, 9), dpi=80, \
                                    facecolor='w', edgecolor='k')
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.062, top=0.95)

           
            axs[0].plot(t_list,theta,label=r'$\theta$',alpha=0.8) 
            axs[0].plot(t_list,theta_dot,label=r'$\dot{\theta}$',alpha=0.8)
            axs[0].plot(t_list,Cphi,label=r'$\cos(\phi)$',alpha=0.8) 
            axs[0].plot(t_list,Sphi,label=r'$\sin(\phi)$',alpha=0.8) 
            axs[0].plot(t_list,phi_dot,label=r'$\dot{\phi}$',alpha=0.8)
            axs[0].set_ylabel(r'states')
            axs[0].margins(x=0)
            axs[0].grid()
            axs[0].legend()
            

        
            axs[1].plot(t_list,u_list,label=r'$u$') #input
            temp = axs[1].twinx()
            trigger=np.array(trigger)* 0.02
            temp.fill_between(t_list,trigger,color='red',alpha=0.4)

           
            temp.plot(t_list,tau_list,color='red',label=r'$\tau$') #input
            temp.legend()
            temp.set_ylim(0,0.3)
            axs[1].set_ylabel(r'action')
            axs[1].margins(x=0)
            axs[1].grid()
            axs[1].legend(bbox_to_anchor=(0.2,1))

            
            
            plt.show()
        return

    def sim_paper(self,IsPlot=True,num=200):
        self.load()
        self.noise = self.STCActionNoise(std_deviation=[0,0])
        prev_state = self.env.reset(IsUp=False,x0=[0,0,-np.pi,0])
        self.env.set_noise(False)        
        episodic_reward = 0

        x_list,u_list,tau_list,t_list  = [self.env.get_state_norm()],[0],[0],[0]
        cost_list,stagecost_list,trigger= [0],[0],[0]
        t = 0
        while True:
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state)
            if(IsPlot):
                print(action)
            stage_cost = - self.beta   #reward for STC (including communication cost)
            last_t = t
            # Recieve state and reward from environment.
            while(t - last_t <= float(action[1])):
                #env.render()
                state, reward, done, info = self.env.step([action[0]])
                
                stage_cost += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))

                t += self.env.dt
                t_list.append(t)
                u_list.append(action[0])
                tau_list.append(action[1])
                x_list.append(self.env.get_state_norm())

                stagecost_list.append(-stage_cost)
                cost_list.append(-episodic_reward)
                
                if(done):
                    break
            trigger.append(t)

            
            episodic_reward += stage_cost * np.exp(- self.alpha * last_t)

            # End this episode when `done` is True
            if done:
                break
            prev_state = state
        
        x_ep,u_ep,tau_ep,cost_ep,com_ep=[],[],[],[],[]
        for i in range(num):
            prev_state = self.env.reset(IsUp=False,x0=[0,0,-np.pi,0])
            self.env.set_noise(True)        
            episodic_reward = 0
            com_cost =0

            x_list1,u_list1,tau_list1  = [self.env.get_state_norm()],[0],[0]
            cost_list1,com_list1= [0],[0]
            t = 0
            while True:
                
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state)
                stage_cost = - self.beta   #reward for STC (including communication cost)
                last_t = t
                
                com_cost += self.beta * np.exp(- self.alpha * last_t)
                # Recieve state and reward from environment.
                while(t - last_t <= float(action[1])):
                    #env.render()
                    state, reward, done, info = self.env.step([action[0]])
                    
                    stage_cost += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))

                    t += self.env.dt
                    u_list1.append(float(action[0]))
                    tau_list1.append(float(action[1]))
                    x_list1.append(self.env.get_state_norm())

                    cost_list1.append(-episodic_reward)
                    com_list1.append(com_cost)
                    if(done):
                        break
                
                
                episodic_reward += stage_cost * np.exp(- self.alpha * last_t)

                # End this episode when `done` is True
                if done:
                    break
                prev_state = state
            if(len(x_list1)!=len(x_list)):
                print(i,'failed')
                continue
            x_ep.append(x_list1)
            u_ep.append(u_list1)
            tau_ep.append(tau_list1)
            cost_ep.append(cost_list1)
            com_ep.append(com_list1)
            print(i,'done')
        x_ep = np.array(x_ep)
        u_ep = np.array(u_ep)
        tau_ep = np.array(tau_ep)
        cost_ep = np.array(cost_ep)
        com_ep = np.array(com_ep)
        perform_ep = cost_ep - com_ep

        ave_cost=np.average(cost_ep,0)
        ave_com=np.average(com_ep,0)
        ave_perform = np.average(perform_ep,0)

        if(IsPlot == True):
            print('Cost=',ave_cost[-1])
            print('performance=',ave_perform[-1])
            print('com=',ave_com[-1])
            #plot with latex style
            #plt.rc('text', usetex=True) #use latex
            fig, axs = plt.subplots(3, 1, figsize=(8, 9), dpi=80, \
                                    facecolor='w', edgecolor='k')
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.062, top=0.95)

            axs[0].fill_between(t_list,np.min(x_ep,0),np.max(x_ep,0),color='b',alpha=0.2)
            #for i in range(len(x_ep)):
            #    axs[0].plot(t_list,x_ep[i],color='b',alpha=0.01)
            axs[0].plot(t_list,x_list,label=r'$\Vert x\Vert_2$') 
            
            axs[0].set_ylabel(r'state')
            axs[0].margins(x=0)
            axs[0].grid()
            axs[0].legend()

            axs[1].fill_between(t_list,np.min(u_ep,0),np.max(u_ep,0),color='b',alpha=0.2)
            #for i in range(len(x_ep)):
            #    axs[1].plot(t_list,u_ep[i],color='b',alpha=0.01)
            axs[1].plot(t_list,u_list,label=r'$u$') #input
            axs[1].set_ylim(-9,9)
            temp = axs[1].twinx()
            trigger=np.array(trigger)
            markerline, stemlines, baseline = temp.stem(trigger,np.ones_like(trigger)*0.02,linefmt='red',basefmt='red')
            markerline.set_markerfacecolor('red')
            markerline.set_markeredgecolor('none')
            # for i in range(len(x_ep)):
            #     temp.plot(t_list,tau_ep[i],color='r',alpha=0.01)
            temp.fill_between(t_list,np.min(tau_ep,0),np.max(tau_ep,0),color='r',alpha=0.2)
            temp.plot(t_list,tau_list,color='red',label=r'$\tau$',linestyle='--') #input
            temp.legend()
            temp.set_ylim(0,0.3)
            axs[1].set_ylabel(r'action')
            axs[1].margins(x=0)
            axs[1].grid()
            axs[1].legend(bbox_to_anchor=(0.2,1))
            # for i in range(len(x_ep)):
            #     axs[2].plot(t_list,cost_ep[i],color='r',alpha=0.01)
            # for i in range(len(x_ep)):
            #     axs[2].plot(t_list,com_ep[i],color='g',alpha=0.01)
            axs[2].fill_between(t_list,np.min(cost_ep,0),np.max(cost_ep,0),color='b',alpha=0.2)
            axs[2].plot(t_list,ave_cost,label=r'$V_{\rm ep}^\pi (x)$',color='b')
            axs[2].fill_between(t_list,np.min(perform_ep,0),np.max(perform_ep,0),color='r',alpha=0.2)
            axs[2].plot(t_list,ave_perform,label=r'Performance Cost',color='r',linestyle='--') 
            axs[2].fill_between(t_list,np.min(com_ep,0),np.max(com_ep,0),color='g',alpha=0.2)
            axs[2].plot(t_list,ave_com,label=r'Communication Cost',color='darkgreen',linestyle='-.')

            
            axs[2].set_ylabel(r'episode cost')
            axs[2].set_xlabel(r'simulation time $t$ [sec]')
            axs[2].set_ylim(0,40)
            axs[2].margins(x=0)
            axs[2].grid()
            axs[2].legend()
            
            plt.show()
        return ave_cost[-1],ave_perform[-1],ave_com[-1]

    def get_ouput_distribution(self,num=21):
        self.load()
        x1 = np.linspace(-np.pi/2,np.pi/2,num)    
        x2 = np.linspace(-np.pi,np.pi,num)  
        state0_list = []

        for i in range(num):
            for j in range(num):
                state0_list.append([x1[i],0,np.cos(x2[j]),np.sin(x2[j]),0])
        state0_list = np.array(state0_list)


        x1, x2 = np.meshgrid(x1, x2)
        u_batch, tau_batch = self.actor(state0_list)
        u_batch = np.array(u_batch)
        tau_batch = np.array(tau_batch)

    
        fig = plt.figure(figsize=(8, 8), facecolor='w')
        
            
        ax_tau = fig.gca(projection='3d')
        #ax_tau.set_title(r'$\tau(x|\omega^\tau)$')
        ax_tau.scatter3D(x1, x2, tau_batch.reshape(num,num).T, color='blue', alpha=0.8)
        ax_tau.set_xlabel(r'$\theta$ [rad]')
        ax_tau.set_ylabel(r'$\phi$ [rad]')
        ax_tau.set_zlabel(r'$\tau(x|\omega^\tau)$ [sec]')
        ax_tau.plot_surface(x1, x2, tau_batch.reshape(num,num).T, color='blue',alpha=0.4)
    


        plt.show()

    def sim_episode(self,IsUP=False):
        self.env.set_noise(True)
        self.noise = self.STCActionNoise(std_deviation=[0,0])
        self.load()
        prev_state = self.env.reset(IsUp=IsUP)

        episodic_reward = 0

        tf_state0 = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        t = 0
        while True:
            
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = self.policy(tf_prev_state)
            stage_cost = - self.beta   #reward for STC (including communication cost)
            last_t = t
            # Recieve state and reward from environment.
            while(t - last_t <= float(action[1])):
                #env.render()
                state, reward, done, info = self.env.step([action[0]])
                
                stage_cost += self.env.dt * reward * np.exp(- self.alpha * (t - last_t))

                t += self.env.dt
        
                if(done):
                    break  
            episodic_reward += stage_cost * np.exp(- self.alpha * last_t)

            # End this episode when `done` is True
            if done:
                break
            prev_state = state
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action0 = self.actor(tf_state0)
        action1 = self.actor(tf_prev_state)
        eva_epi_cost = self.critic((tf_state0,action0[0],action0[1])) - np.exp(-self.alpha * 10) * self.critic((tf_prev_state,action1[0],action1[1]))
        return float(-episodic_reward),float(-eva_epi_cost)

    def Vep_graph(self,filename=None,num=100,split=10,IsUp=True,scale=[10,11]):
        if filename is None:
            Vep_list = []
            for i in range(num):
                Vep_list.append([self.sim_episode(IsUP=IsUp)])
                print('{}/{}:{}'.format(i,num,Vep_list[-1]))
            Vep_list = np.array(Vep_list)
            pickle.dump(Vep_list,open(self.path+'VepList.veplist','wb'))
        else:
            Vep_list = pickle.load(open('{}.veplist'.format(self.path+filename),'rb'))
        
        X,Y = Vep_list[:,0,0],Vep_list[:,0,1]
        Xmax,Xmin,Ymax,Ymin = max(X),min(X),max(Y),min(Y)
        XA= Xmax-Xmin
        YA= Ymax-Ymin
        Vep = np.linspace(Xmin,Xmax,split)
        Qep = np.linspace(Ymin,Ymax,split)
        V,Q = np.meshgrid(Vep,Qep)
        P = np.zeros((split,split))
        for data in Vep_list:
            P[int((data[0,0]-Xmin)/XA*(split-1)),int((data[0,1]-Ymin)/YA*(split-1))] +=1
        P = P.T

        ct = plt.contourf(V,Q,P/(num),cmap='binary',levels=10,vmin=0.004,vmax=0.02)
        plt.plot([scale[0],scale[1]],[scale[0],scale[1]],color='black',linestyle='--',linewidth=1)
        plt.xlabel(r'$V_{\rm ep}^\pi(x)$')
        plt.ylabel(r'$V_{\rm ep}(x)$')
        cb = plt.colorbar()
        if IsUp == True:
            plt.title(r'$P(V_{\rm ep}^\pi,V_{\rm ep}|x(0)\sim d^{\rm up}_0)$')
        else:
            plt.title(r'$P(V_{\rm ep}^\pi,V_{\rm ep}|x(0)\sim d^{\rm down}_0)$')      
        plt.show()

def check_tau0():
    for i in range(30):
        tau= (i+1)*0.01
        stc = DDPG4STC(tau0=tau,path='./stctest/')
        #stc.train_start()
        print(float(stc.actor(np.array([0,0,0,0,0]).reshape((1,5)))[1]))
        del stc

def graph_tau0():
    import time
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)
    for i in range(8):  
        tau= (i+22)*0.01
        stc = DDPG4STC(tau0=tau,path='./stctest/{}/'.format(i+22))
        stc.beta = 0.4
        stc.algorithm1(int(tau*100000))
        del stc
        print('#### DONE ##### tau=',tau)
        localtime = time.asctime(time.localtime(time.time()))
        print(localtime)

def simulation(name,num=0):
    stc = DDPG4STC(path='./stctest/{}/'.format(name))
    stc.noise = stc.STCActionNoise(std_deviation=[0,0])
    stc.simulation(IsZeroInput=True,IsNoise=False,x0=[0,0,-np.pi/3,0])

def sim_compare(num=20):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), dpi=80, \
                                facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.13, top=0.95)
    t_list, ave_cost, ave_perform, ave_com = [],[],[],[]
    for i in range(17):
        stc = DDPG4STC(path='./stctest/{}/'.format(i+5))
        stc.beta=1
        stc.noise = stc.STCActionNoise(std_deviation=[0,0])
        cost,perform,com = stc.sim_paper(IsPlot=False, num=num)
        t_list.append((i+5)*0.01)
        ave_cost.append(cost)
        ave_perform.append(perform)
        ave_com.append(com)
        del stc
    axs.plot(t_list,ave_cost,label=r'$V_{\rm ep}^\pi (x,200)$',color='b')
    axs.plot(t_list,ave_perform,label=r'Performance Cost',color='r') 
    axs.plot(t_list,ave_com,label=r'Communication Cost',color='darkgreen')
    axs.grid()
    axs.legend()
    plt.show()


if __name__ == '__main__':   
    #check_tau0()
    #graph_tau0()
    #simulation(23)
    #sim_compare()
    stc = DDPG4STC(tau0=0.1,path='./new/{}/'.format(2))
    stc.algorithm1(20000)
    #del stc
    #stc = DDPG4STC(problem='RotaryPend-v1',tau0=0.1,path='./new/{}/sdc/'.format(2))
    #stc.load()
    #stc.Vep_graph('down',num=50000,split=50,IsUp=False,scale=[16,33])
    #stc.sim_paper(num=200)
    #stc.get_ouput_distribution(num=21)
    del stc
    # print('#### DONE ##### tau=',0.1)
    #stc.noise = stc.STCActionNoise(std_deviation=[0,0])
    #stc.simulation(IsZeroInput=False,IsNoise=False,x0=[0,0,-np.pi,0])
    # del stc 
    pass