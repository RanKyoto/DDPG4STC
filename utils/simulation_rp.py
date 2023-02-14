import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ddpg4stc import SelfDDPG4STC
from time import sleep

def simulation(agent:SelfDDPG4STC,IsPlot=True,x0=[0,0,-np.pi,0],IsNoise=True,IsZeroInput=False):
    ''' simulation for the rotary inverted pendulum'''
    agent.noise = np.array([0,0])
    prev_state = agent.env.reset(x0=x0)
    agent.env.set_noise(IsNoise)        
    episodic_reward = 0
    theta,theta_dot, Cphi, Sphi, phi_dot = [x0[0]],[x0[1]],[np.cos(x0[2])],[np.sin(x0[2])],[x0[3]]
    u_list,tau_list,t_list  = [0],[0],[0]
    cost_list,stagecost_list,trigger= [0],[0],[1]
    t = 0

    while True:    
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = agent.policy(tf_prev_state,IsExploration=False)

        stage_cost = - agent.beta   #reward for STC (including communication cost)
        last_t = t
        # Recieve state and reward from environment.
        while(t - last_t <= float(action[1])):
            agent.env.render()
            state, reward, done, info = agent.env.step([action[0]])
            sleep(0.001)
            stage_cost += agent.env.dt * reward * np.exp(- agent.alpha * (t - last_t))

            t += agent.env.dt
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
   
        episodic_reward += stage_cost * np.exp(- agent.alpha * last_t)

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