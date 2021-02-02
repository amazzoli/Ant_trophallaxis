# TEST WITH MD
import numpy as np
from Ants_env import AntsEnv

best_policy = np.array([[1-0.5, 1-0.990728, 1-0.995043, 1-0.995836, 1-0.996288, 1-0.996002, 1-0.99514, 1-0.992916, 1-0.977079, 1-0.000989989, 1-0.000114517], #prob share
                        [0.5, 0.980344, 0.984584, 0.987594, 0.97947,  0.942716, 0.552266, 0.499074, 0.784932, 0.951355, 0.00964495]])   #prob take

#best_policy = np.array([[1-0.5, 1-0.990728, 1-0.995043, 1-0.995836, 1-0.996288, 1-0.996002, 1-0.99514, 1-0.992916, 1-0.977079, 1-0.000989989, 1-0.000114517], #prob share
#                        [0.5, 0.980344, 0.984584, 0.987594, 0.97947,  0.942716, 0.552266, 0.499074, 0.5, 0.5, 0.00964495]])   #prob take




#best_policy = 1 - np.array([[ 0.6739599, 1.0, 1.0, 0.99999994, 1.0, 1.0, 0.99999994, 1.0, 0.9999999, 0.20892602, 0.13199264], #prob share
#                        [0.06267894, 0, 0, 0, 0, 0, 0, 0, 0.2264, 0.27290568, 0.34064603]])   #prob pass

n_MD = 4000
av_reg = 0

for iMD in range(n_MD):

        # Initialize AntEnv        
        Ants = AntsEnv(Nr=1, Mmax=10, c=0.1, rg=0.1, gamma=0.995, ran_init=True)
        state = Ants.get_state()
        
        init = [False for i in range(2)]
        active = state[1] 
        init[active] = True
        
        
        done = False
        rewards = np.zeros(2) 
        tot_time = 0
        tot_rew = np.zeros(2)
        # -----------------------------------------

        while not done:
        
            #action = agents[active].get_actions() #return actions vector to give particles, and label
            action = np.int( np.random.rand() <= best_policy[active, state[2+active]])

            state, rews, done, time = Ants.step(action) #evolve systems from given actions

            tot_time += time
            rewards += rews
            active = state[1]

            #print('{} {} {} {} {} {} {}\n'.format(old_state, state, rewards, action, done, time, tot_time))          
            
            if  not init[active]:
                rewards[active] = 0
                init[active] = True
            else:
                if not done:
                    tot_rew[active] += rewards[active]
                    rewards[active] = 0
                else:
                    for i, ini in enumerate(init):
                        if ini:
                            tot_rew[i] += rewards[i]
        
        av_reg += tot_rew
        print('traj length {} rew {} {}'.format(tot_time, *(av_reg/(iMD+1))))
