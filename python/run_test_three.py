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
best_policy = 1- np.array([[0.36619538, 0.96754366 ,0.98490757, 0.9790372, 0.9785732, 0.9479718, 0.8741505, 0.813861, 0.43110576, 0.30929458, 0.22816245], #prob gather 
                          [ 0.49206212, 0.02621105, 0.01831029, 0.025723431, 0.049529675, 0.09136002, 0.19391663, 0.16478524, 0.15067384, 0.38373724, 0.49482346],
                          [0.36619538, 0.96754366, 0.98490757 ,0.9790372, 0.9785732, 0.9479718, 0.8741505, 0.813861, 0.43110576, 0.30929458, 0.22816245]])   #prob pass

N = 3
n_MD = 4000
av_reg = np.zeros(N)

for iMD in range(n_MD):

        # Initialize AntEnv        
        Ants = AntsEnv(Nr=2, Mmax=10, c=0.1, rg=0.1, gamma=0.995, ran_init=True, deathpenalty=2, eatreward=0)
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
