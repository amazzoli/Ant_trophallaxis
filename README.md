# Ant trophallaxis

### Instructions for a basic training:

- Compile the code with
g++ -o run.exe run.cpp lib/alg.cpp lib/ants.cpp lib/nac.cpp lib/utils.cpp -std=c++17

- Write the parameters from the notebook write_parameters.ipynb. This generate the parameter file in data/system_name

- Run the training from run.exe. One should specify also two string which are the system name (which model is used, e.g. ant_cons) and the run name (which set of parameters are used). Exemple:
./run.exe ant_cons 2p_nac5

- Import the trajectories and make the plots from analize_training.ipynb