# Analysis of food distribution.
import numpy as np

data_dir = ['NOCONSUME_3p_gamma0.9', 'NOCONSUME_3p_gamma0.95401367', 'NOCONSUME_3p_gamma0.97885257', 'NOCONSUME_3p_gamma0.99027508/',  'NOCONSUME_3p_gamma0.99552786/', 'NOCONSUME_3p_gamma0.99794343/', 'NOCONSUME_3p_gamma0.99905426/', 'NOCONSUME_3p_gamma0.99956509/', 'NOCONSUME_3p_gamma0.9998/', 'NOCONSUME_3p_gamma1/']

data_dir = ['NOCONSUME_2p_gamma0.9', 'NOCONSUME_2p_gamma0.95401367', 'NOCONSUME_2p_gamma0.97885257', 'NOCONSUME_2p_gamma0.99027508/',  'NOCONSUME_2p_gamma0.99552786/', 'NOCONSUME_2p_gamma0.99794343/', 'NOCONSUME_2p_gamma0.99905426/', 'NOCONSUME_2p_gamma0.99956509/', 'NOCONSUME_2p_gamma0.9998/', 'NOCONSUME_2p_gamma1/']

gamma = [0.9, 0.95401367, 0.97885257, 0.99027508, 0.99552786, 0.99794343, 0.99905426, 0.99956509, 0.9998, 0.9999]
coco = open('dead_F', "w")

for gam in gamma:
    dir = './NOCONSUME_3p_gamma'+str(gam)+'_pFilling0.5_GREEDY'
    dir = './NOCONSUME_3p_gamma'+str(gam)

dir = './'
Ndata = 1000000
data = np.loadtxt(dir + '/ev_info.txt', skiprows=1, max_rows=Ndata)
N = (data.shape[1]-2) // 4

Nrecipients = N - 1
Mmax = 19
forager_gathering_food = np.array([])
agents = np.array([])
food_exch = np.array([])
episodes = np.sum(data[:,-2])

time_episode = np.append([0], np.cumsum(data[:,-2]*data[:,-1]))[:-1]
time = data[:,-1] + time_episode
time_spent = time - np.append([0], time)[:-1]

food_forager = data[:,0]%(Mmax+1)
food_colony = np.mean(data[:,1:N]%(Mmax+1), axis=1)
food_forager_after = data[:,N*2]%(Mmax+1)
food_colony_after = np.mean(data[:,N*2+1:N*3]%(Mmax+1), axis=1)

change_food = food_colony_after - food_colony
np.savetxt('food_F.txt', food_forager.reshape(-1,1))
np.savetxt('food_F_after.txt', food_forager_after.reshape(-1,1))
np.savetxt('food_R.txt', np.append( np.append(time.reshape(-1,1), food_colony_after.reshape(-1,1), axis = 1), data[:,-1].reshape(-1,1), axis=1) )

rewardF = np.sum(data[:,N*3]) / episodes
rewardR = [np.sum(data[:,N*3+1+i]) for i in range(Nrecipients)] / episodes




print(gam, rewardF, rewardR)

dead = np.logical_or(data[:,N*2:N*3] == 0, data[:,N*2:N*3] == Mmax + 1)
deadF = [np.sum(dead[:,0]) / episodes]
deadR = [ np.sum(dead[:,1+i])/episodes  for i in range(Nrecipients)]


print(deadF)
# Fractions of accepts per colony health.
agents = np.argmax( data[:,:N] < Mmax, axis=1)
actions = data[range(agents.shape[0]),N+agents]
accept = np.logical_and(agents>0, actions==0)
reject = np.logical_and(agents>0, actions==1)

foraging = np.logical_and(agents==0, actions==0)
not_foraging = np.logical_not(foraging)

food_accept, _ = np.histogram(food_colony[accept], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
histo_colony, _ = np.histogram(food_colony[np.logical_not(dead[:,0])], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
histo_change, _ = np.histogram(food_colony[np.logical_not(dead[:,0])])
food_reject, _ = np.histogram(food_colony[reject], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
time_colony, _ = np.histogram(food_colony, bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent)

time_forager_out, _ = np.histogram(food_colony[foraging], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent[foraging])
time_forager_in, _ = np.histogram(food_colony[not_foraging], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent[not_foraging])

np.savetxt(dir+'/accept_ratio.txt', food_accept/(food_accept + food_reject))
np.savetxt(dir+'/inflow.txt', food_accept/time_colony)
np.savetxt(dir+'/time_histo.txt', time_colony)
np.savetxt(dir+'/time_in_out.txt', np.append(time_forager_out, time_forager_in, axis=0).reshape(-1,2))

gathering_events = np.logical_and(data[:,0] < Mmax+1, data[:,N] == 0)
food_gather = data[gathering_events, 0]
forager_gathering_time = data[gathering_events, -1]
new_gathering_events = (np.append([-10], forager_gathering_time)[:-1] - forager_gathering_time != -1)
forager_gathering_food = food_gather[new_gathering_events]

new_agent = (np.append([-1], agents)[:-1] != agents)
count_act = -1

food_exch = np.array([])

while (new_agent.shape[0] > 0) and ((np.argmax(new_agent) or new_agent[0])) :
    split_index = np.argmax(new_agent) 
    _, new_agent = np.array_split(new_agent, [split_index+1])
    if agents[count_act] > 0 and actions[count_act]==0:
        food_avail = np.min([data[count_act,0]%11, data[count_act, agents[count_act]]])
        food_exch = np.append(food_exch, split_index)
    count_act += split_index + 1

#print(final_food_colony, final_food_forager)


#food_dist_r, _ = np.histogram(final_colony,  bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
#food_dist_f, _ = np.histogram(final_forager, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_gath, _ = np.histogram(forager_gathering_food, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_exch, _ = np.histogram(food_exch, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))

#food_dist_r = food_dist_r / (count_ep*N)
#food_dist_f = food_dist_f / count_ep
food_dist_gath = food_dist_gath / forager_gathering_food.shape[0]
food_dist_exch = food_dist_exch / food_exch.shape[0]

#for i in range(Mmax+1):
#    #print('{:.5f} {:.5f} {:.5f} {:.5f}'.format(food_dist_r[i], food_dist_f[i], food_dist_gath[i], food_dist_exch[i]))
#    print('{:.10f} {:.10f}'.format(food_dist_gath[i], food_dist_exch[i]))

coco.write(dir[18:]+' {} {}\n'.format(*deadF, rewardF))

