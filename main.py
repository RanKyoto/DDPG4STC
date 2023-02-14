"""
Title: Deep Reinforcement Learning for Continuous-Time 
       Self-Triggered Control with Experimental Evaluation
Author: Ran,Wang
Date created: 2023/02/14
Last modified: 2022/02/14
Description: Implementing Deep Reinforcement Learning for
             Continuous-time Self-triggered Control (DDPG4STC)
"""
import os
from ddpg4stc import DDPG4STC
from utils.simulation_rp import simulation

os.environ['CUDA_VISIBLE_DEVICES']='3' # use GPU

def training():
    agent = DDPG4STC(name='demo')
    agent.algorithm_ptc(Ne=20000)
    agent.algorithm_stc(Ne=20000)

def demo():
    agent = DDPG4STC(problem= "RotaryPend-v1",name='demo')
    agent.load(version='ptc')
    simulation(agent=agent)

if __name__ == '__main__':   
    demo()
    pass