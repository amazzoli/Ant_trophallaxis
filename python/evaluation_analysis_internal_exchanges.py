# Analysis of food distribution.
import numpy as np

# ---------------
dir = './'
Ndata = 1000000
data = np.loadtxt(dir + '/ev_info.txt', skiprows=1, max_rows=Ndata)
N = (data.shape[1]-2) // 4
print(N)
Nrecipients = N - 1
Mmax = 19
# ---------------


episodes = np.sum(data[:,-2])
end_episodes = (data[:,-2]==1)
filled_colony = np.sum(np.sum(data[end_episodes,N*2+1:N*3]%(Mmax+1)) == Mmax*Nrecipients)
print('#Sated = {}, #Episodes = {}, #Sated/#Ep = {}%'.format(filled_colony, episodes, filled_colony / episodes *100))

# Fractions of accepts per colony health.

# Active agent in colony (0 if nobody)
agents_colony = np.argmax( data[:,:N] < Mmax, axis=1)
# Active agent in trophallaxis (0 if nobody)
agents_tropha = np.argmax(np.logical_and(data[:,1:N]>=Mmax, data[:,1:N]<2*Mmax), axis=1)

colony_state = np.any(data[:,:N] < Mmax, axis=1)
tropha_state = np.any(np.logical_and(data[:,1:N]>=Mmax, data[:,1:N]<2*Mmax))

internal_exch = np.any(data[:,1:N] >= 3*Mmax)
external_exch = np.logical_and(tropha_state, np.logical_not(internal_exch) )

# Active agent irrespective to what/where/why
any_agents = np.maximum(agents_colony,agents_tropha)
actions = data[range(any_agents.shape[0]),N+any_agents]
accept = np.logical_and(agents_tropha>0, actions==0)
reject = np.logical_and(agents_tropha>0, actions==1)

external_exch_accept = np.logical_and(accept, external_exch)
external_exch_reject = np.logical_and(reject, external_exch)

# Forager gathering
foraging = np.logical_and(agents_colony==0, actions==0)
not_foraging = np.logical_not(foraging)
# Forager initiating trophallaxis 
feeding_in = np.logical_and(agents_colony==0, actions==1)

time_episode = np.append([0], np.cumsum(data[:,-2]*data[:,-1]))[:-1]
time = data[:,-1] + time_episode
time_spent = time - np.append([0], time)[:-1]
# Time passes only if forager is involved, either foraging or trying trophallaxis.
time_spent[any_agents > 0]=0


food_forager = data[:,0]%(Mmax+1)
food_colony = np.mean(data[:,1:N]%(Mmax+1), axis=1)
food_forager_after = data[:,N*2]%(Mmax+1)
food_colony_after = np.mean(data[:,N*2+1:N*3]%(Mmax+1), axis=1)

change_food = food_colony_after - food_colony
np.savetxt('food_F.txt', food_forager.reshape(-1,1))
np.savetxt('food_F_after.txt', food_forager_after.reshape(-1,1))
np.savetxt('food_R.txt', np.append( np.append(time.reshape(-1,1), food_colony_after.reshape(-1,1), axis = 1), data[:,-1].reshape(-1,1), axis=1) )
np.savetxt('food_R1.txt', (data[:,1]%(Mmax+1)).reshape(-1,1))
np.savetxt('food_R2.txt', (data[:,2]%(Mmax+1)).reshape(-1,1))

rewardF = np.sum(data[:,N*3]) / episodes
rewardR = [np.sum(data[:,N*3+1+i]) for i in range(Nrecipients)] / episodes
print(rewardF, rewardR)

dead = (data[end_episodes,N*2:N*3]%(Mmax + 1)==0)
deadF = [np.sum(dead[:,0]) / episodes]
deadR = [ np.sum(dead[:,1+i])/episodes  for i in range(Nrecipients)]

print(deadF, deadR)

food_accept, _ = np.histogram(food_colony[external_exch_accept], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
histo_colony, _ = np.histogram(food_colony[np.logical_not(dead[:,0])], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
histo_change, _ = np.histogram(food_colony[np.logical_not(dead[:,0])])
food_reject, _ = np.histogram(food_colony[external_exch_reject], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
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

new_agent = (np.append([-1], any_agents)[:-1] != any_agents)
count_act = -1

food_exch = np.array([])

while (new_agent.shape[0] > 0) and ((np.argmax(new_agent) or new_agent[0])) :
    split_index = np.argmax(new_agent) 
    _, new_agent = np.array_split(new_agent, [split_index+1])
    if any_agents[count_act] > 0 and actions[count_act]==0:
        food_avail = np.min([data[count_act,0]%11, data[count_act, any_agents[count_act]]])
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

print(dir[18:]+' {} {}\n'.format(*deadF, rewardF))

