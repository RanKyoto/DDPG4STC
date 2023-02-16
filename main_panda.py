import sys,os,warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import gymnasium 
sys.modules["gym"] = gymnasium
import panda_gym
import numpy as np
from ddpg4stc import DDPG4STC_Panda
from utils.simulation_panda import simulation


tasks = ["PandaReachDense-v3","PandaPushDense-v3","PandaSlideDense-v3",
        "PandaPickAndPlaceDense-v3","PandaStackDense-v3"]

def training(task='PandaReachDense-v3'):
    '''training an STC policy for a panda task'''

    os.environ['CUDA_VISIBLE_DEVICES']='0' # use GPU
    agent = DDPG4STC_Panda(problem= task,
                    learning_rate=[0.01,0.001,0.0005],
                    name='demo',render=False,beta=1.0)
    #agent.algorithm_ptc(Ne=1000)
    agent.algorithm_stc(Ne=1000,retraining=True)


# def training_all():
#     for task in tasks:
#         print(task)
#         print("----------------------")
#         training(task=task)

def testing_all():
    cost_list = [] 
    for task in tasks:
        cost = demo('stc',show=False)
        cost_list.append(cost)
    for i in range(5):
        print(tasks[i],cost_list[i])

def demo(Type='stc',task='PandaReachDense-v3',show = True):
    '''
        testing the learned controllers

        Type=  'stc' (self-triggered controller)  

        Type = 'ptc' (periodic-triggered controller)
    '''
    agent = DDPG4STC_Panda(problem= task,
                        name='demo',
                        render=True,
                        beta=1.0)
    agent.load(version=Type,IsLoadReplay=False)
    cost = simulation(agent=agent,seed=315)
    return cost

if __name__ == '__main__':
    print(tasks[0])
    print("----------------------")
    #training(tasks[0])
    cost = demo('stc',task=tasks[0])
    print(cost)