import gymnasium 
import panda_gym,sys
sys.modules["gym"] = gymnasium

from ddpg4stc import DDPG4STC

problem = 'PandaReachDense-v3'

agent = DDPG4STC(problem=problem)
agent.algorithm_ptc(agent)