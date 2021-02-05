# TEST WITH MD
import numpy as np
from Ants_env import AntsEnv

#best_policy = np.array([[1-0.5, 1-0.990728, 1-0.995043, 1-0.995836, 1-0.996288, 1-0.996002, 1-0.99514, 1-0.992916, 1-0.977079, 1-0.000989989, 1-0.000114517], #prob share
#                        [0.5, 0.980344, 0.984584, 0.987594, 0.97947,  0.942716, 0.552266, 0.499074, 0.784932, 0.951355, 0.00964495]])   #prob take

#best_policy = np.array([[1-0.5, 1-0.990728, 1-0.995043, 1-0.995836, 1-0.996288, 1-0.996002, 1-0.99514, 1-0.992916, 1-0.977079, 1-0.000989989, 1-0.000114517], #prob share
#                        [0.5, 0.980344, 0.984584, 0.987594, 0.97947,  0.942716, 0.552266, 0.499074, 0.5, 0.5, 0.00964495]])   #prob take




best_policy = 1- np.array([[0.9258067, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.63132393], #prob gather 
                          [0.009211875, 0, 0, 0, 0, 0, 0, 0, 0.0184299, 0.25760314, 0.36691833],
                          [0.004089072, 0, 0, 0, 0, 0, 0, 0, 0.0079019, 0.23968118, 0.29052776]])   #prob pass

# DEATH PENALTY (-1) - NO EAT FOR RECEIVERS.
best_policy = 1- np.array([[0.34896883, 0.32698694, 0.7263123, 0.77450866, 0.8509365, 0.9458485, 0.9266954, 0.8644445, 0.4863153, 0.25367025, 0.07568344], #prob gather 
                          [0.23756112, 0.0973868, 0.06790047, 0.055291682, 0.07438586, 0.074849375, 0.12828878, 0.11427487, 0.082441956, 0.08534909, 0.5123963],
                          [0.7373114, 0.1575939, 0.053952005, 0.12503375, 0.10713754, 0.08000285, 0.10050939, 0.08770655, 0.086167544, 0.16340484, 0.47413573]])   #prob pass

N = 3
n_MD = 4000
av_reg = 0

for iMD in range(n_MD):

        # Initialize AntEnv        
        Ants = AntsEnv(Nr=N-1, Mmax=10, c=0.1, rg=0.1, gamma=0.995, ran_init=True)
        state = Ants.get_state()
        
        init = [False for i in range(N)]
        active = state[1] 
        init[active] = True
        
        
        done = False
        rewards = np.zeros(N) 
        tot_time = 0
        tot_rew = np.zeros(N)
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
        print('traj length {} rew {} {} {}'.format(tot_time, *(av_reg/(iMD+1))))
