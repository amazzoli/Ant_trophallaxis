# TEST WITH MD
import numpy as np
from Ants_env import AntsEnv
from single_agent import Agent
import scipy
import time

# ------------
n_input = 1
n_actions = 2
# ------------

# ------------
forager = Agent()
receiver = Agent()
# ----
agents = [forager, receiver]
# -----------

n_MD = 200

for iMD in range(n_MD):

        # Initialize AntEnv        
        Ants = AntsEnv(Nr=1, Mmax=10, c=0.01, rg=0.05, gamma=0.99)
        state = Ants.get_state()
        active = state[1]
        agents[active].initialize(state[2+active])
        print(state)
        
        not_init = True
        done = False
        rewards = np.zeros(2) 
        # -----------------------------------------
        while not done:
            active = state[1]
            action = agents[active].get_actions() #return actions vector to give particles, and label
            state, rews, done = Ants.step(action) #evolve systems from given actions
            rewards += rews
            
            if  state[1] == 1 and not_init:
                rewards[1] = 0
                agents[1].initialize(state[2+1])
                not_init = False
            
            if not done:
                print(state, rewards, action, done)
                rewards[active] = 0
                agents[active].add_env_timeframe(state[2+active], rewards[active], done)
            else:
                print(state, rewards, action, done)
                for i, a in enumerate(agents):
                    a.add_env_timeframe(state[2+i], rewards[i], done)

        for a in agents:
            a.train_step(epochs=20)

Agent.save_models(path=models_rootname, final_save = True)



