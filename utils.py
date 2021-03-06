import numpy as np
import matplotlib.pyplot as plt
import os


def p_law_burn(x, x_burn, expn, c0, cc):
    """Power law function with a burn-in period"""
    if x < x_burn:
        return c0
    else:
        return c0*cc / (cc + (x-x_burn)**expn)


def plot_lr(alg_params):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(7,3))

    ax1.set_xlabel('Learning step', fontsize=14)
    ax1.set_ylabel('Critic learning rate', fontsize=14)
    ax1.set_yscale('log')
    xs = np.linspace(0, alg_params['n_steps'], 100)
    lr = [p_law_burn(x, alg_params['a_burn'], alg_params['a_expn'], alg_params['a0'], alg_params['ac']) for x in xs]
    ax1.plot(xs, lr)
    ax2.set_xlabel('Learning step', fontsize=14)
    ax2.set_ylabel('Actor learning rate', fontsize=14)
    ax2.set_yscale('log')
    eps = [p_law_burn(x, alg_params['b_burn'], alg_params['b_expn'], alg_params['b0'], alg_params['bc']) for x in xs]
    ax2.plot(xs, eps)
    return fig, (ax1, ax2)


def write_params(param_dict, dir_path, file_name):
    """Write a parameter file"""
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print ("Creation of the directory failed")
    f = open(dir_path + file_name, "w")
    for k,v in param_dict.items():
        if type(v) is list or type(v) is np.ndarray:
            f.write(k + "\t")
            for i in range(len(v)):
                f.write(str(v[i])+",")
            f.write("\n")
        else:
            f.write(k + "\t" + str(v) + "\n")
    f.close()
    

def read_params(path):
    """Read a parameter file"""
    params = dict()
    f = open(path, "r")
    for l in f.readlines():
        try:
            params[l.split()[0]] = float(l.split()[1])
        except ValueError:
            if ',' not in l.split()[1]:
                params[l.split()[0]] = l.split()[1]
            else:
                try:
                    params[l.split()[0]] = np.array(l.split()[1].split(',')[:-1], dtype=float)
                except ValueError:
                    params[l.split()[0]] = np.array(l.split()[1].split(',')[:-1])
    return params

    
def read_traj(path, header=True):
    """Read a trajectory with headers"""
    f = open(path, "r")
    v_traj = []
    if header:
        state_labels = f.readline().split()
    for line in f.readlines():
        v_traj.append(line.split())
    if header:
        return np.array(v_traj, dtype=float), state_labels
    else:
        return np.array(v_traj, dtype=float)
    

def read_2d_traj(path):
    """Read a two dimensional trajectory"""
    f = open(path, "r")
    header = np.array([v.split(',')[:-1] for v in f.readline().split()])
    traj = []
    for line in f.readlines():
        traj_at_time = []
        for elem in line.split():
            traj_at_state = np.array(elem.split(',')[:-1], dtype=float)
            traj_at_time.append(traj_at_state)
        traj.append(traj_at_time)
    return traj, header


def smooth_traj(traj, wind_size):
    """Binned average over the trajectory, with bin of size wind_size.
    It returns the binned x-axis and the averaged y-axis"""
    new_traj, times = [], []
    i = 0
    while i < len(traj)-wind_size:
        new_traj.append(np.mean(traj[i:i+wind_size]))
        times.append(i + wind_size/2)
        i += wind_size
    return times, new_traj



    
    
    