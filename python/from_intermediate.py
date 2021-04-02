import numpy as np
for i in range(3):
    coco = open('policy'+str(i)+'_traj.txt').readlines()[7]
    momo = coco.replace(',','\t').split()
    lolo = np.array([float(i) for i in momo])
    
    with open('new_policy'+str(i)+'.txt', "w") as file:
        
        for c in range(42):
            if c<21:
                file.write('{}\t{}\t\n'.format(lolo[2*c], lolo[2*c+1]))
            else:
                file.write('1.0\t\n')

