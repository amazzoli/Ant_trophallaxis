# Analysis of food distribution.
import numpy as np
data_dir = './'

Ndata = 1000000
data = np.loadtxt(data_dir+'/ev_info.txt', skiprows=Ndata//2+1, max_rows=Ndata)

N = (data.shape[1]-2) // 4
print(N)
Nrecipients = N - 1
#N = Nrecipients + 1
Mmax = 10
histo = np.zeros(Mmax+1)

forager_gathering_food = np.array([])
agents = np.array([])
food_exch = np.array([])

rewardF = [np.mean(data[:,N*3]), np.std(data[:,N*3])/np.sqrt(data.shape[0]) ]
rewardR = [ [np.mean(data[:,N*3+1+i]) , np.std(data[:,N*3+1+i])/np.sqrt(data.shape[0])]  for i in range(Nrecipients)]

dead = np.logical_or(data[:,:N] == 0, data[:,:N] == Mmax + 1)
deadF = [np.mean(dead[:,0]) , np.std(dead[:,1:N])/np.sqrt(dead.shape[0])]
deadR = [ [np.mean(dead[:,1+i]) , np.std(dead[:,1+i])/np.sqrt(dead.shape[0])]  for i in range(Nrecipients)]

agents = np.argmax( data[:,:N] < N, axis=1)
actions = data[range(agents.shape[0]),N+agents]

gathering_events = np.logical_and(data[:,0] < Mmax+1, data[:,N] == 0)
food_gather = data[gathering_events, 0]
forager_gathering_time = data[gathering_events, -1]
new_gathering_events = (np.append([-10], forager_gathering_time)[:-1] - forager_gathering_time != -1)
forager_gathering_food = food_gather[new_gathering_events]

new_agent = (np.append([-1], agents)[:-1] != agents)
count_act = 0

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

for i in range(Mmax+1):
    #print('{:.5f} {:.5f} {:.5f} {:.5f}'.format(food_dist_r[i], food_dist_f[i], food_dist_gath[i], food_dist_exch[i]))
    print('{:.5f} {:.5f}'.format(food_dist_gath[i], food_dist_exch[i]))

print("rewards for F and R")
print(*rewardF, "\n", *rewardR)
print("death for F and R")
print(*deadF, "\n", *deadR)
print("death for F and R")
print(*deadF, "\n", *deadR)
print(deadF[0], np.mean(np.array(deadR)[:,0]), rewardF[0], np.mean(np.array(rewardR)[:,0]))
    

