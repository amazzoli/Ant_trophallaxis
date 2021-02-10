# Analysis of food distribution.
import numpy as np
data = np.loadtxt(data_dir+'/ev_info.txt', skiprows=1)

N = 11
Mmax = 10
histo = np.zeros(Mmax+1)

count_ep = 0

final_colony = np.array([])
final_forager = np.array([])

while np.argmax(data[:,4*N]==1) :
    count_ep += 1
    split_index = np.argmax(data[:,4*N]==1)
    ep, data = np.array_split(data, [split_index+1])
    agent = np.argmax( ep[:,:N]<N)
    final_colony = np.append(final_colony, ep[-1,1:N]%(Mmax+1))
    final_forager = np.append(final_forager, ep[-1,0]%(Mmax+1))
    final_food_colony = np.sum(ep[-1,1:N]%11)
    final_food_forager = ep[-1,0]%11
    #print(final_food_colony, final_food_forager)

food_dist_r, _ = np.histogram(final_colony,  bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))
food_dist_f, _ = np.histogram(final_forager, bins=(Mmax+1), range=(-0.5, (Mmax+0.5)))