"""
Title: Deep Reinforcement Learning for Continuous-Time 
       Self-Triggered Control with Experimental Evaluation
Author: Ran,Wang
Date created: 2023/02/14
Last modified: 2023/02/14
Description: Implementing Deep Reinforcement Learning for
             Continuous-time Self-triggered Control (DDPG4STC)
"""

import os
from tkinter import N
from ddpg4stc import DDPG4STC
from utils.simulation_rp import simulation

def training():
    '''training an STC policy for a rotary inverted pendulum'''

    os.environ['CUDA_VISIBLE_DEVICES']='1' # use GPU
    agent = DDPG4STC(problem= "RotaryPend-v0",name='demo')
    agent.algorithm_ptc(Ne=20000)
    agent.algorithm_stc(Ne=20000)

def demo(Type='stc',Disturbunce=False):
    '''
        testing the learned controllers

        Type=  'stc' (self-triggered controller)  

        Type = 'ptc' (periodic-triggered controller)
    '''
    agent = DDPG4STC(problem= "RotaryPend-v0",name='demo')
    agent.load(version=Type,IsLoadReplay=False)
    simulation(agent=agent,IsNoise=Disturbunce)

if __name__ == '__main__':
    '''Step 1 : Trainging a STC policy'''
    # os.environ['CUDA_VISIBLE_DEVICES']='1' # use GPU   
    # training()
    
    # '''Step 2 : Show the simulation results'''
    demo(Type='stc',Disturbunce=False)
    pass