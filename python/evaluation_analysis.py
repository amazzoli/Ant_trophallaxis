# Analysis of food distribution.
import numpy as np
data_dir = './'
data = np.loadtxt(data_dir+'/ev_info.txt', skiprows=1)

N = 11
Mmax = 10
histo = np.zeros(Mmax+1)

count_ep = 0

final_colony = np.array([])
final_forager = np.array([])
forager_gathering_food = np.array([])
agents = np.array([])
food_exch = np.array([])

while (np.argmax(data[:,4*N]==1) or (data[0,4*N]==1)) :
    count_ep += 1
    split_index = np.argmax(data[:,4*N]==1)
    ep, data = np.array_split(data, [split_index+1])
    agents = np.argmax( ep[:,:N] < N, axis=1)
    actions = ep[range(agents.shape[0]),N+agents]
    
    final_colony = np.append(final_colony, ep[-1,(2*N+1):3*N]%(Mmax+1))
    final_forager = np.append(final_forager, ep[-1,2*N]%(Mmax+1))
    final_food_colony = np.sum(ep[-1,(2*N+1):3*N]%11)
    final_food_forager = ep[-1,2*N]%11
    
    gathering_events = np.logical_and(ep[:,0] < Mmax+1, ep[:,N] == 0)
    forager_gathering_food = np.append(forager_gathering_food, ep[gathering_events, 0])
    
    new_agent = (np.append([-1], agents)[:-1] != agents)
    count_act = 0
    while (new_agent.shape[0] > 0) and ((np.argmax(new_agent) or new_agent[0])) :
        split_index = np.argmax(new_agent) 
        _, new_agent = np.array_split(new_agent, [split_index+1])
        if agents[count_act] > 0 and actions[count_act]==1:
            food_avail = np.min([ep[count_act,0]%11, ep[count_act, agents[count_act]]])
            food_exch = np.append(food_exch, split_index)
        count_act += split_index + 1
    #print(final_food_colony, final_food_forager)

food_dist_r, _ = np.histogram(final_colony,  bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_f, _ = np.histogram(final_forager, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_gath, _ = np.histogram(forager_gathering_food, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_exch, _ = np.histogram(food_exch, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))

food_dist_r = food_dist_r / (count_ep*N)
food_dist_f = food_dist_f / count_ep
food_dist_gath = food_dist_gath / forager_gathering_food.shape[0]
food_dist_exch = food_dist_exch / food_exch.shape[0]

for i in range(Mmax+1):
    print('{:.5f} {:.5f} {:.5f} {:.5f}'.format(food_dist_r[i], food_dist_f[i], food_dist_gath[i], food_dist_exch[i]))

