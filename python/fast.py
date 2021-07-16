import numpy as np
import sys

#if __name__ == "__main__":
gamma = 1
dir = './'
Ndata = 1000000
data = np.loadtxt(dir + '/ev_info.txt', skiprows=1, max_rows=Ndata)
N = (data.shape[1]-2) // 4

Nrecipients = N - 1
Mmax = 19
episodes = np.sum(data[:,-2])
dead = np.logical_or(data[:,N*2:N*3] == 0, data[:,N*2:N*3] == Mmax + 1)
deadF = [np.sum(dead[:,0]) / episodes]
deadR = [ np.sum(dead[:,1+i])/episodes  for i in range(Nrecipients)]

momo = np.loadtxt('return_traj.txt', skiprows=1)
print(gamma, np.mean(momo, axis=0)[1], np.std(momo,axis=0)[1]/np.sqrt(len(momo[:,1])), *deadF)

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
print('average food crop at exit', np.mean( data[foraging, 0] ))