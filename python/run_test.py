# TEST WITH MD
import numpy as np
from Ants_env import AntsEnv


def policy(state):
    if state[1]==1 and state[3]<=2:
        return 0
    elif state[1]==0 and state[2]<=4:
        return 0
    return 1

n_MD = 100

for iMD in range(n_MD):

        # Initialize AntEnv        
        Ants = AntsEnv(Nr=1, Mmax=10, c=0.01, rg=0.05, gamma=0.99)
        state = Ants.get_state()
        print(state)
        
        not_init = True
        done = False
        rewards = np.zeros(2) 
        tot_time = 0
        # -----------------------------------------
        while not done:
            #active = state[1]
            #action = agents[active].get_actions() #return actions vector to give particles, and label
            action = policy(state)
            state, rews, done, time = Ants.step(action) #evolve systems from given actions
            tot_time += time
            rewards += rews
            active = state[1]

            if  state[1] == 1 and not_init:
                rewards[1] = 0
                not_init = False
            
            if not done:
                print(state, rewards, action, done, time, tot_time)
                rewards[active] = 0
                #agents[active].add_env_timeframe([], state[2+active], rewards[active])
            else:
                print(state, rewards, action, done, time, tot_time)
                #for a, i in enumerate(agents):
                    #a.add_env_timeframe([], state[2+i], rewards[i])
        print('traj length {}'.format(tot_time))
#        Agent.train_step(epochs=20)

#Agent.save_models(path=models_rootname, final_save = True)

