import tensorflow as tf
from tensorflow.keras.layers import Dense,Concatenate,Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import numpy as np

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