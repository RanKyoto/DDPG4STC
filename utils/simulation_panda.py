import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ddpg4stc import SelfDDPG4STC_Panda
from time import sleep

def norm_state(observation):
    return np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])

def simulation(agent:SelfDDPG4STC_Panda,IsPlot=True,x0=[0,0,-np.pi,0],seed:int=-1):
    ''' simulation for the rotary inverted pendulum'''
    if seed <0:
        observation, info = agent.env.reset()
    else:
        observation, info = agent.env.reset(seed=seed)
    old_state = agent.state_extract(observation)
    episodic_cost = 0
    com_cost = agent.beta
    state_list = [norm_state(observation)]
    u_list,tau_list,t_list  = [0],[0],[0]
    cost_list,trigger= [0],[1]
    t = 0

    while t < 10.0:    
        u,tau = agent.policy(np.expand_dims(old_state,0),IsExploration=False)

        stage_cost = agent.beta   #reward for STC (including communication cost)
        n_substeps = int(tau/agent.env.sim.timestep)
        agent.env.sim.n_substeps=n_substeps
        
        observation, reward, terminated, truncated, info  = agent.env.step(u)
        for i in range(n_substeps):
            t += agent.env.sim.timestep
            stage_cost += -reward* np.exp(-agent.alpha* i* agent.env.sim.timestep) 
            cost_list.append(episodic_cost)
            t_list.append(t)
            u_list.append(np.linalg.norm(u))
            tau_list.append(tau)
            state_list.append(norm_state(observation))
            trigger.append(0)
        episodic_cost += stage_cost* np.exp(-agent.alpha*t)
        com_cost += agent.beta * np.exp(-agent.alpha*t)
        trigger[-1] = 1
        trigger[-2] = 1

        old_state = agent.state_extract(observation)
    

    if(IsPlot == True):
        #plot with latex style
        #plt.rc('text', usetex=True) #use latex
        fig, axs = plt.subplots(2, 1, figsize=(8, 9), dpi=80, \
                                facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.062, top=0.95)

        
        axs[0].plot(t_list,state_list,label=r'$|x|$',alpha=0.8) 
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
    return episodic_cost, com_cost