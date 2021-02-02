# TEST WITH MD
import numpy as np
from Ants_env import AntsEnv
from single_agent import Agent
import scipy
import time
from tqdm import tqdm



def from_policy_to_prob(o, policy):
    logp = policy(o)
    prob = np.exp(logp)
    prob = prob / np.sum(prob)
    return prob

def obs(crop):
    o = np.zeros((1,11))
    o[0,crop] = 1
    return o

# ------------
n_input = 11
n_actions = 2
# ------------

# ------------

restart = False

forager = Agent(input_dim=1, output_dim=2, models_rootname='./forager', restart_models= restart)
receiver = Agent(input_dim=1, output_dim=2, models_rootname='./receiver', restart_models= restart)
# ----
agents = [forager, receiver]
# -----------

n_MD = 5000

for iMD in tqdm(range(n_MD)):

        # Initialize AntEnv        
        done = False
        Ants = AntsEnv(Nr=1, Mmax=10, c=0.1, rg=0.1, gamma=0.995, ran_init=True)
        state = Ants.get_state()
        old_state = np.zeros(state.shape)
        old_state[:] = state
        active = state[1] 

        # initial state
        # Since only active states are considered
        # Agents for receiver ants could be un-initialized for whole episode. 
        init = [False for i in range(2)]
        
        agents[active].initialize(obs(state[2+active]))    
        init[active] = True
        
        rewards = np.zeros(2) 
        
        tot_time = 0
        tot_rewards = np.zeros(2)
        
        count = 0
        # -----------------------------------------
        with open('traj_data/traj{}.dat'.format(iMD), "w") as ftraj: 
        
            while not done:
                count += 1
                action = agents[active].get_actions() #return actions vector to give particles, and label
                
                old_state[:] = state
                state, rews, done, time = Ants.step(action) #evolve systems from given actions

                tot_time += time
                rewards += rews
                
                active = state[1]

                ftraj.write('{} {} {} {} {} {} {}\n'.format(old_state, state, rewards, action, done, time, tot_time))          
                
                if  not init[active]:
                    rewards[active] = 0
                    agents[active].initialize(obs(state[2+active]))
                    init[active] = True
                else:
                    if not done:
                        tot_rewards[active] += rewards[active]
                        agents[active].add_env_timeframe(obs(state[2+active]), rewards[active], done)
                        rewards[active] = 0
                    
                    else:
                        for i, a in enumerate(agents):
                            if init[i]:
                                a.add_env_timeframe(obs(state[2+i]), rewards[i], done)
                                tot_rewards[i] += rewards[i]



        print('policy f: ', *[from_policy_to_prob(np.array([[i]]), agents[0].policy)[0,0] for i in range(11)])
        print('policy r: ', *[from_policy_to_prob(np.array([[i]]), agents[1].policy)[0,0] for i in range(11)])
        print('value f: ', *[agents[0].critic(np.array([[i]]))[0,0].numpy() for i in range(11)])
        print('value r: ', *[agents[1].critic(np.array([[i]]))[0,0].numpy() for i in range(11)])
        
        print('{} traj: time= {} rew= {} {} deadF {} deadR {}'.format(iMD, tot_time, tot_rewards[0], tot_rewards[1], np.int(state[2]<=0), np.int(state[3]<=0)))
        print(init, ' INIT')
        for i,a in enumerate(agents):
            print(init[i])
            if init[i]:
                print('training agent {}: {}'.format(i,a))
                a.train_step(epochs=10)

# save models
for a in agents:
    a.save_models(final_save = True)



