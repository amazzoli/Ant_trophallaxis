# Analysis of food distribution.
import numpy as np
from scipy.stats import binned_statistic
# ---------------
dir = './'
Ndata = 3000000
data = np.loadtxt(dir + '/ev_info.txt', skiprows=1, max_rows=Ndata)
N = (data.shape[1]-2)//4
print(N)
Nrecipients = N - 1
Mmax = 19
# ---------------


episodes = np.sum(data[:,-2])
end_episodes = (data[:,-2]==1)
filled_colony = np.sum(np.sum(data[end_episodes,N*2+1:N*3]%(Mmax+1), axis=1) == Mmax*Nrecipients)
mask = np.logical_and(end_episodes, np.sum(data[:,N*2+1:N*3]%(Mmax+1), axis=1) == Mmax*Nrecipients)
mask = np.logical_and(mask, data[:,0]%(Mmax+1) == 0)

print('#Sated = {}, #Episodes = {}, #Sated/#Ep = {}%'.format(filled_colony, episodes, filled_colony / episodes *100))

# Fractions of accepts per colony health.

# Active agent in colony (0 if nobody)
agents_colony = np.argmax( data[:,:N] < (Mmax+1), axis=1)
# Active agent in trophallaxis (0 if nobody)
second_macrostate = np.logical_and(data[:,:N] >= (Mmax+1), data[:,:N]<2*(Mmax+1))
second_macrostate[:,0] = False
any_agents = np.argmax(second_macrostate, axis=1)
any_agents = np.max( np.append(any_agents.reshape(-1,1), agents_colony.reshape(-1,1), axis=1), axis=1)

# ----------------------------------------
colony_state = np.any(data[:,:N] < (Mmax+1), axis=1)
tropha_state = np.any(np.logical_and(data[:,1:N]>=(Mmax+1), data[:,1:N]<2*(Mmax+1)), axis=1)

actions = data[range(any_agents.shape[0]),N+any_agents]
accept = np.logical_and(any_agents>0, actions==0)
reject = np.logical_and(any_agents>0, actions==1)

# Forager gathering
foraging = np.logical_and(any_agents==0, actions==0)
not_foraging = np.logical_not(foraging)
# Forager initiating trophallaxis 
feeding_in = np.logical_and(any_agents==0, actions==1)
feeds = np.append(np.logical_and(feeding_in[:-1], accept[1:]), False)

time_episode = np.append([0], np.cumsum(data[:,-2]*data[:,-1]))[:-1]
time = data[:,-1] + time_episode
time_spent = time - np.append([0], time)[:-1]
# Time passes only if forager is involved, either foraging or trying trophallaxis.
time_spent[any_agents > 0]=0
time_spent[feeds] = 0

# -------------------------------------------------------
# --- CALCULATION FOOD EXCHANGED IN EVENTS
# -------------------------------------------------------

new_agent = (np.append([-1], any_agents)[:-1] != any_agents)
count_act = 0

food_exch = np.array([])

foodbins=Mmax
food_exch_binned = [np.array([]) for i in range(foodbins)]
food_exch_binned_for = [np.array([]) for i in range(foodbins)]
#norm_exch = np.array([])

while (new_agent.shape[0] > 1) and ((np.argmax(new_agent[1:]) or new_agent[1])) :
    split_index = np.argmax(new_agent[1:])
    #print('\n\n')
    #print('previous data was ', new_agent)    
    garbage, new_agent = np.array_split(new_agent, [split_index+1])  
    #print('remaining data is ', new_agent)
    #print('split_index is{}?'.format(split_index))
    #print('splitted data is ', garbage)
    #print('count_act: ', count_act, 'agent is: ', any_agents[count_act], 'action is ', actions[count_act])
    #if any_agents[count_act] > 0 and actions[count_act]==0 and data[count_act, any_agents[count_act]] > (Mmax+1):
    if any_agents[count_act] > 0 and data[count_act, any_agents[count_act]] > (Mmax+1):
        food_of_rec = data[count_act, any_agents[count_act]]%(Mmax+1)
        food_of_for = data[count_act,0]%(Mmax+1)
        food_of_rec_after = data[count_act+split_index+1, any_agents[count_act]]%(Mmax+1)
        #print('food of rec before{}, food of rec after {}'.format(food_of_rec, food_of_rec_after))
        if (food_of_rec < Mmax):#food_avail = Mmax - data[count_act, any_agents[count_act]]%(Mmax+1)
            #print('recorded')
            fd = split_index
            if (food_of_rec_after == Mmax and actions[count_act+split_index] == 0):
                fd -= 0
            #print('food appended is ', fd)
            food_exch = np.append(food_exch, fd)
            #norm_exch = np.append(norm_exch, split_index/food_avail)
            #iexch = (int(food_of_rec) -1) // (Mmax//foodbins)
            frec = int(food_of_rec)
            food_exch_binned[frec] = np.append(food_exch_binned[frec], fd)
            ffor = int(food_of_for) - 1
            #iexch = (int(food_of_for) -2) // (Mmax//foodbins)
            food_exch_binned_for[ffor] = np.append(food_exch_binned_for[ffor], fd)
            # CHANGES TIME SPENT
            if split_index>0:
                time_spent[count_act:count_act+split_index] = 1/split_index
        
    count_act += split_index+1

for i in range(foodbins):
    print("{} array has {} entries".format(i, food_exch_binned[i].shape))
    print("{} distribution has {} mean and {} standard error".format(i, 1./np.mean(food_exch_binned[i]), 1./(np.mean(food_exch_binned[i])*food_exch_binned[i].shape[0])))

for i in range(foodbins):
    print("{} array has {} entries".format(i, food_exch_binned_for[i].shape))
    print("{} distribution has {} mean and {} standard error".format(i, 1./np.mean(food_exch_binned_for[i]), 1./(np.mean(food_exch_binned_for[i])*food_exch_binned_for[i].shape[0])))


# Time for trophallaxis is divided into equal parts.
time = np.cumsum(time_spent)
time_episode_true = np.zeros(time.shape)
count_time = 0

end_episodes[-1] = True

while (count_time < data.shape[0]):
    split_index = np.argmax(end_episodes) 
    _, end_episodes = np.array_split(end_episodes, [split_index+1])
    if (count_time>0):
        time_episode_true[count_time:count_time+split_index+1] = time[count_time-1]
    count_time += split_index + 1

time_zero = time - time_episode_true


end_episodes = (data[:,-2]==1)
end_episodes[-1] = True

Nbins_ev = 125
average_evolution = np.zeros(Nbins_ev)
average_evolution2 = np.zeros(Nbins_ev)

av_condition = np.zeros(Nbins_ev)
av_count = np.zeros(Nbins_ev)

average_evolution_int = np.zeros(Nbins_ev)
count_time = 0

food_forager = data[:,0]%(Mmax+1)
food_colony = np.mean(data[:,1:N]%(Mmax+1), axis=1)
food_forager_after = data[:,N*2]%(Mmax+1)
food_colony_after = np.mean(data[:,N*2+1:N*3]%(Mmax+1), axis=1)



with open("food_R_zero.txt", "a") as f:
    while (count_time < data.shape[0]):
        split_index = np.argmax(end_episodes) 
        _, end_episodes = np.array_split(end_episodes, [split_index+1])
        if (count_time>0):
            av_ep, _, _ = binned_statistic(np.append(0, time_zero[count_time:count_time+split_index+1]),
                                           np.append(3/19., food_colony_after[count_time:count_time+split_index+1]), 
                                           bins=Nbins_ev, range=(0,5000))
            
            av_condition += np.nan_to_num(av_ep, nan=0)
            av_count += (av_ep>0)
                
            av_ep = np.nan_to_num(av_ep, nan=food_colony_after[count_time+split_index])
            
            average_evolution += av_ep
            average_evolution2 += av_ep*av_ep
            np.savetxt(f, np.append(time_zero[count_time:count_time+split_index+1].reshape(-1,1), food_colony_after[count_time:count_time+split_index+1].reshape(-1,1), axis = 1))
        f.write("\n\n")
        count_time += split_index + 1 

np.savetxt('food_evol_conditional.txt', np.append( (av_condition/av_count).reshape(-1,1), av_count.reshape(-1,1)/episodes, axis=1) )

average_evolution /= episodes
err_av_ev = np.sqrt(average_evolution2 - average_evolution*average_evolution)/np.sqrt(episodes)

food_dist_exch, _ = np.histogram(food_exch, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)), density=True)
np.savetxt(dir+'/dist_exch.txt', food_dist_exch)

food_dist_exch_binned = np.zeros((Mmax+1, 2*foodbins))
food_dist_exch_binned_for = np.zeros((Mmax+1, 2*foodbins))

for i in range(foodbins):
    food_dist_exch_binned[:,i], _ = np.histogram(food_exch_binned[i], bins=(Mmax+1), range=(0, Mmax), density=False)
    food_dist_exch_binned_for[:,i], _ = np.histogram(food_exch_binned_for[i], bins=(Mmax+1), range=(0, Mmax), density=False)


food_dist_exch_binned[:,foodbins:] = np.sqrt(food_dist_exch_binned[:,:foodbins]) / np.sum(food_dist_exch_binned[:,:foodbins], axis = 0, keepdims=True) * Mmax
food_dist_exch_binned[:,:foodbins] *= Mmax / np.sum(food_dist_exch_binned[:,:foodbins], axis = 0, keepdims=True)

food_dist_exch_binned_for[:,foodbins:] = np.sqrt(food_dist_exch_binned_for[:,:foodbins]) / np.sum(food_dist_exch_binned_for[:,:foodbins], axis = 0, keepdims=True) * Mmax
food_dist_exch_binned_for[:,:foodbins] *= Mmax / np.sum(food_dist_exch_binned_for[:,:foodbins], axis = 0, keepdims=True)


np.savetxt(dir+'/dist_exch_binned_all.txt', np.nan_to_num(food_dist_exch_binned, 0.))
np.savetxt(dir+'/dist_exch_binned_forager_all.txt', np.nan_to_num(food_dist_exch_binned_for, 0.))


#food_dist_exch_norm, _ = np.histogram(norm_exch, bins=(Mmax+1)//2, range=(-0.5/Mmax, (Mmax+0.5)/Mmax), density=True)
#np.savetxt(dir+'/dist_exch_norm.txt', food_dist_exch_norm)

# --------------------------------------------------------


change_food = food_colony_after - food_colony
np.savetxt('food_F.txt', food_forager.reshape(-1,1))
np.savetxt('food_F_after.txt', food_forager_after.reshape(-1,1))
np.savetxt('food_R.txt', np.append( np.append(time.reshape(-1,1), food_colony_after.reshape(-1,1), axis = 1), data[:,-1].reshape(-1,1), axis=1) )
#np.savetxt('food_R_zero.txt', np.append( np.append(time_zero.reshape(-1,1), food_colony_after.reshape(-1,1), axis = 1), data[:,-1].reshape(-1,1), axis=1) )
np.savetxt('food_evol_average.txt', np.append(average_evolution.reshape(-1,1), err_av_ev.reshape(-1,1), axis=1))
np.savetxt('food_R1.txt', (data[:,1]%(Mmax+1)).reshape(-1,1))
np.savetxt('food_R2.txt', (data[:,2]%(Mmax+1)).reshape(-1,1))

rewardF = np.sum(data[:,N*3]) / episodes
rewardR = [np.sum(data[:,N*3+1+i]) for i in range(Nrecipients)] / episodes
print(rewardF, rewardR)

dead = (data[end_episodes,N*2:N*3]%(Mmax + 1)==0)
deadF = [np.sum(dead[:,0]) / episodes]
deadR = [ np.sum(dead[:,1+i])/episodes  for i in range(Nrecipients)]

print(deadF, deadR)

#Nbins = (Mmax*Nrecipients)
Nbins = 25
food_accept, _ = np.histogram(food_colony[accept], bins=Nbins, range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) , weights=time_spent[accept])
histo_colony, _ = np.histogram(food_colony_after[end_episodes], bins=Nbins, range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) )
food_reject, _ = np.histogram(food_colony[reject], bins=Nbins, range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ) , weights=time_spent[reject])
time_colony, _ = np.histogram(food_colony, bins=Nbins, range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent)

time_forager_out, _ = np.histogram(food_colony[foraging], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent[foraging])
time_forager_in, _ = np.histogram(food_colony[not_foraging], bins=(Mmax*Nrecipients), range=(-0.5/(Mmax*Nrecipients), Mmax+0.5/(Mmax*Nrecipients) ), weights=time_spent[not_foraging])

np.savetxt(dir+'/accept_ratio.txt', food_accept/(food_accept + food_reject))
np.savetxt(dir+'/inflow.txt', food_accept/time_colony)
np.savetxt(dir+'/final_food_colony.txt', histo_colony)

np.savetxt(dir+'/food_accept.txt', food_accept)
np.savetxt(dir+'/time_histo.txt', time_colony)
np.savetxt(dir+'/time_in_out.txt', np.append(time_forager_out.reshape(-1,1), time_forager_in.reshape(-1,1), axis=1))

gathering_events = np.logical_and(data[:,0] < Mmax+1, data[:,N] == 0)
food_gather = data[gathering_events, 0]
forager_gathering_time = data[gathering_events, -1]
new_gathering_events = (np.append([-10], forager_gathering_time)[:-1] - forager_gathering_time != -1)
forager_gathering_food = food_gather[new_gathering_events]


# INFLOW
momo = np.append([0], average_evolution)
momo = np.append(momo, [0])
diff = momo[2:]-momo[:-2]
diff[-1] = 0
diff /= 2
diff[0] = momo[2] - momo[1]
err = np.append([0], err_av_ev)
err = np.append(err, [0])
err = err[2:]+err[:-2]
err[0] += err_av_ev[0]

np.savetxt('inflow.txt' , np.concatenate((average_evolution.reshape(-1,1), diff.reshape(-1,1), err.reshape(-1,1)), axis=1 ))
