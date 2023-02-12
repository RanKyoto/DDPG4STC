"""
Edited by Ran, Wang
Quanser Rotary Inverted Pendulum with Wiener Process Disturbunce
Final Edit Date: 2023.02.12
History:
1.Fix some mistakes about disturbunce calculation. 
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class RotaryPendulumEnv(gym.Env):
    """
    Description:
        Note: This is a continuious-time input version of RotaryPendulumEnv
        A pole is attached by an un-actuated joint to a rotary arm. 

    Source:
        This environment corresponds to the version of Quanser's production

    Observation:
        Type: Box(4)
        Num     Observation                     Min                 Max
        0       Rotary Arm angle                - pi/2 rad          pi/2 rad
        1       Rotary Arm Angular Velocity     -Inf                Inf
        2       Pole Angle                      - pi rad            pi rad
        3       Pole Angular Velocity           -Inf                Inf

    Actions:
        Type: Continuous(1)
        Num   Action
        Voltage [-voltage_mag,voltage_mag]

    Reward:
        Stage cost is defined like optimal control problem, 
        which consider both control performance and energy cost.

    """

    def __init__(self):
        self.g  = 9.81    
        # Gravitational constant (m/s^2)
        K_IN2M  = 0.0254 
        # from Inch to Meter    
        K_OZ2N = 0.2780139  
        # from oz-force to N
        K_N2OZ = 1 / K_OZ2N 
        # from N to oz-force
        K_RDPS2RPM = 60 / ( 2 * np.pi ) 
        # from rad/s to RPM
        '''Parameters of 12-inch Pendlulum'''
        self.Mp = 0.127  
        # Pendulum Mass with T-fitting (kg)
        self.Lp = (13 + 1 / 4) * K_IN2M 
        # Pendulum Full Length (with T-fitting, from axis of rotation to tip) (m)
        #self.lp = (6 + 1 / 8) * K_IN2M  
        # Distance from Pivot to Centre Of Gravity: calculated experimentally (m)
        self.Jp = self.Mp * (self.Lp ** 2) /12  
        # Pendulum Moment of Inertia (kg.m^2) - approximation (kg.m^2)
        self.Bp = 0.0024                        
        # Equivalent Viscous Damping Coefficient (N.m.s/rad)
        '''Parameters of Rotary Arm'''
        self.Mr = 0.257                         
        # Arm Mass with T-fitting (kg)
        self.Lr = 8.5 * K_IN2M                  
        # Full Length of Arm (from axis of rotation to tip) (m)
        # self.lr = (2 + 7 / 16) * K_IN2M         
        # Distance from Pivot to Centre Of Gravity: calculated experimentally (m)
        self.Jr = self.Mr * (self.Lr ** 2) /12  
        # Arm Moment of Inertia (kg.m^2) - approximation (kg.m^2)
        self.Br = 0.0024                        
        # Equivalent Viscous Damping Coefficient (N.m.s/rad)
        '''Parameters of Servo Motor'''
        self.Rm = 2.6 
        # Armature Resistance (Ohm)
        self.kt = 1.088 * K_OZ2N * K_IN2M  
        # = .00767  Motor Torque Constant (N.m/A)
        # M_e_max = 0.566 * K_OZ2N * K_IN2M 
        # = 0.566 oz.in (parameter not used) Continuous torque (N.m/A)
        self.km = 0.804 / 1000 * K_RDPS2RPM 
        # = .00767 Motor Back-EMF Constant (V.s/rd)
        self.Kg = 70 
        # Internal Gear Ratio (of the Planetary Gearbox)
        self.eta_g = 0.90 
        # Gearbox Efficiency
        self.eta_m = 0.69 
        # Motor ElectroMechanical Efficiency
        '''Constant Terms for Kinematic'''
        self.C1 = self.Mp * (self.Lr ** 2) + 1/4 * self.Mp * (self.Lp**2)
        self.C2 = 1/4 * self.Mp * (self.Lp ** 2)
        self.C3 = 1/2 * self.Mp * self.Lp * self.Lr
        self.C4 = 2 * self.C2
        self.C5 = self.C3
        self.C6 = self.eta_g * self.Kg * self.eta_m * self.kt / self.Rm
        self.C7 = self.Kg * self.km
        self.C8 = self.C3
        self.C9 = self.Jp + self.C2
        self.C10= self.C2
        self.C11= 1/2 * self.Mp * self.Lp * self.g
        '''Noise Enable'''
        self.W_en = 0
        ''' State Space Representation '''
        Jt = self.Jr*self.Jp + self.Mp*((self.Lp/2)**2)*self.Jr + self.Jp*self.Mp*(self.Lr**2)
        #!!!Note that the Related state should be [theta,phi,theta_dot,phi_dot],
        #!!!which is different form the sequence of self.state
        A =np.array( [[0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, (self.Mp**2)*((self.Lp/2)**2)*self.Lr*self.g/Jt, -self.Br*(self.Jp+self.Mp*((self.Lp/2)**2))/Jt, -self.Mp*(self.Lp/2)*self.Lr*self.Bp/Jt],
            [0, self.Mp*self.g*(self.Lp/2)*(self.Jr+self.Mp*(self.Lr**2))/Jt, -self.Mp*(self.Lp/2)*self.Lr*self.Br/Jt, -self.Bp*(self.Jr+self.Mp*self.Lr**2)/Jt]])

        B = np.array([0, 0, (self.Jp+self.Mp*(self.Lp/2)**2)/Jt, self.Mp*(self.Lp/2)*self.Lr/Jt])
        #Add actuator dynamics
        A[2,2] = A[2,2] - self.Kg**2*self.kt*self.km/self.Rm*B[2]
        A[3,2] = A[3,2] - self.Kg**2*self.kt*self.km/self.Rm*B[3]
        B = self.Kg * self.kt * B / self.Rm
        self.A = A.reshape(4,4)     
        self.B = B.reshape(4,1)

        #Configuration
        self.voltage_mag = 8.0 # max input voltage magnitude 
        self.dt = 0.005  # seconds between state updates
        self.kinematics_integrator = "euler"
 
        self.alpha_threshold_radians = np.pi
        self.theta_threshold = np.pi/2

        # Observation bound:
        high = np.array(
            [
                self.theta_threshold * 2,np.finfo(np.float32).max, 1,1,np.finfo(np.float32).max 
            ],  #theta, theta_dot, cos(phi), sin(phi), phi_dot
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action bounds:
        self.action_space = spaces.Box(
            low=-self.voltage_mag,
            high=self.voltage_mag,
            shape=(1,),
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_noise(self,EnNosie=True):
        if EnNosie:
            self.W_en=1
        else:
            self.W_en=0
    
    def get_state_norm(self):
        return np.linalg.norm(self.state)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        theta, theta_dot, alpha, alpha_dot = self.state
        #Edited by Ran, Wang
        Vm = np.clip(action, -self.voltage_mag, self.voltage_mag)[0]
        # Ctheta = math.cos(theta)
        # Stheta = math.sin(theta)
        Calpha = math.cos(alpha)
        Salpha = math.sin(alpha)

        # For the interested reader:
        # https://www.quanser.com/products/rotary-inverted-pendulum/
        # write the nonlinear equations as
        # ax + by = e
        # cx + dy = f
        # then x = (de-bf)/(ad-bc)    x ~ thetaacc  
        #      y = (af-ce)/(ad-bc)    y ~ alphaacc
        
        a = self.C1 - self.C2 * (Calpha ** 2) + self.Jr
        b = - self.C3 * Calpha
        e = - self.C4 * Salpha * Calpha * theta_dot * alpha_dot -  self.C5 * Salpha * (alpha_dot ** 2) +\
            self.C6 * (Vm- self.C7 * theta_dot) - self.Br * theta_dot
        c = - self.C8 * Calpha
        d = self.C9
        f = self.C10 * Calpha * Salpha * (theta_dot ** 2) + self.C11 * Salpha - self.Bp * alpha_dot
        tmp = a * d - b * c
        thetaacc = (d * e - b * f)/tmp +  self.W_en * self.acc_noise(variance=4)
        alphaacc = (a * f - c * e)/tmp +  self.W_en * self.acc_noise(variance=16)

        if self.kinematics_integrator == "euler":
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * thetaacc
            alpha = alpha + self.dt * alpha_dot
            alpha_dot = alpha_dot + self.dt * alphaacc
        else:  # semi-implicit euler
            theta_dot = theta_dot + self.dt * thetaacc
            theta = theta + self.dt * theta_dot
            alpha_dot = alpha_dot + self.dt * alphaacc
            alpha = alpha + self.dt * alpha_dot
            
        self.state = (theta, theta_dot, angle_normalize(alpha), alpha_dot)

        done = bool(
            theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        #Edited by Ran, Wang
        cost = 2 * (theta ** 2) + 4 * (angle_normalize(alpha) ** 2)  + 0.01 * (Vm ** 2)\
            + 0.1 * (theta_dot ** 2) + 0.05 * (theta_dot ** 2)
        if(done):
            cost += 500 
        return self._get_obs(), -cost, done, {}

    def reset(self,IsUp=False ,x0 = None):
        if x0 is not None:
            self.state = np.array(x0)
            return self._get_obs()

        if(IsUp):
            random_position = np.random.uniform(-0.5,0.5)
            high = np.pi/15
            random_angle = np.random.uniform(-high,high)
        else:
            random_position = np.random.uniform(-1,1)
            high = np.pi/3
            random_angle = angle_normalize(np.pi + np.random.uniform(-high,high))
        self.state = np.array([random_position,0,random_angle,0])
        return self._get_obs()
    
    def obs_noise(self,variance = [0.01,0.04,0.01,0.09]):#
        '''Observation noise is represented as White Noise variance should be a np.array '''
        variance = np.array(variance)
        return np.random.normal(np.zeros(np.shape(variance)),variance)

    def acc_noise(self,variance = 1):
        '''Disturbance is represented as Wiener Process variance = sigma^2 '''
        return np.random.normal(0,variance*self.dt)

    def _get_obs(self):
        '''
        theta       Rotary Arm angle                - pi/4 rad          pi/4 rad
        theta_dot   Rotary Arm Angular Velocity     -Inf                Inf
        alpha       Pole Angle                      - pi rad            pi rad
        alpha_dot   Pole Angular Velocity           -Inf                Inf
        '''
        theta, theta_dot, alpha, alpha_dot = self.state + self.W_en * self.obs_noise()
        return np.array([theta,theta_dot, np.cos(alpha), np.sin(alpha),alpha_dot], dtype=np.float32) 

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 600
        center_x = screen_width/2
        center_y = screen_height/2

        world_width = self.theta_threshold * 2
        scale = screen_width / world_width
        carty = 300  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (4 * self.Lp)
        armwidth = 10.0
        armlen  = scale * (2 * self.Lr)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = (
                -armwidth / 2,
                armwidth / 2,
                armlen - armwidth / 2,
                -armwidth / 2,
            )
            arm = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            arm.set_color(0.8, 0.5, 0.5)
            self.armtrans = rendering.Transform(translation=(center_x,center_y + armlen),rotation=np.pi)
            arm.add_attr(self.armtrans)
            self.viewer.add_geom(arm)

            self.motor = rendering.make_circle(13)
            self.motor.set_color(0, 0, 0)
            self.motortrans = rendering.Transform(translation=(center_x,center_y + armlen),scale=(1.3,0.9))
            self.motor.add_attr(self.motortrans)
            self.viewer.add_geom(self.motor)


            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.9, 0.9, 0.9)
            self.viewer.add_geom(self.axle)



            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        armrot = np.arctan(x[0]*scale/armlen)
        self.armtrans.set_rotation(np.pi + armrot)
        self.armtrans.set_scale(1,1/np.cos(armrot))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi