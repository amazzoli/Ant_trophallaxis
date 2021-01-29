import numpy as np

class AntsEnv():

    def __init__(self, Nr=1, Mmax=10, c=0.01, rg=0.05, gamma=0.99):
        # contains initialization data
        self.done = False
        # initial state initialization
        self.Nr = Nr
        self.N = 1 + self.Nr
        self.Mmax = Mmax
        # state is M/m (MACRO/micro state), ID_active, c_forager, c_recipient1, c_recipient2, ...   
        self.state = np.zeros(3+Nr, dtype='int')
        self.state[:2] = 0
        self.state[2] = self.Mmax
        self.state[3:] = self.Mmax//2
        self.alive = np.ones(1+Nr)
        self.c = c
        self.rg = rg
        self.gamma = gamma
        # length of last step of sharing
        self.share = 1
        
    def get_state(self):
        return self.state
    
    def step(self, action):
        # needs to
        # apply change to internal state
        # calculate rewards
        # returns observation, rewards, done_flag
        self.rewards = np.zeros(self.N)
        # Food is consumed AFTER a gathering and AFTER a sharing.
        if not bool(self.state[0]):
            # MACRO(0) state
            # possible actions are Share(1) / Gather(0)
            
            # Share(1)
            if bool(action):
                # Results:
                # Change of state -------
                # - from MACRO to micro
                self.state[0] = 1
                # Recipient ID randomly chosen
                self.state[1] = 1 + np.random.randint(self.Nr)
                self.share = 1
                # No rewards.
                # No time elapsed - End of World, Consumption: none.

            # Gather(0)
            else :
                # Results:
                # Change of state ------
                # Evaluation gathering time for success.
                time = np.random.geometric(p=self.rg)
                # End of World is evalued.
                endtime = np.random.geometric(p=1-self.gamma)
                self.done = ( endtime <= time )
                
                if not self.done:
                    # Consumption.
                    eaten = np.random.binomial(np.min([endtime, time]), self.c, size=self.N)
                    self.state[-self.N:] -= eaten
                    # Death condition is checked for all.
                    self.alive -= (self.state[-self.N:] == 0).astype(int)
                
                    if np.any([self.alive[-self.N:-self.N+1] == 0, np.all(self.alive[-self.N+1:]==0)]):
                        self.done = True
                        
                    # gathering successfull.
                    # c_forager to max
                    self.state[-self.N] = self.Mmax
                    # No Rewards --------------

        else:
            # micro(1) state
            # possible actions are Take(1) / Pass(0)
            # if receiver is full and Takes, action is changed into pass.
            #
            if bool(action) and self.state[2+self.state[1]] == self.Mmax:
                self.share += 1
                action = 0
            
            if bool(action):
                # Take()
                # Change of state ------
                # Forager loses 1
                # Recipient gains 1
                rec = 2 + self.state[1]
                if (self.state[rec] < self.Mmax):
                    # Rewards --------------
                    self.state[2] -= 1
                    self.state[rec] += 1
                    self.rewards[0] += 1 # Forager rewarded.
                    self.rewards[self.state[1]] += 1 # Receiver rewarded.
                    
                self.share += 1 # Length of sharing episode.
                
                # Death condition is checked for all.
                self.alive -= (self.state[-self.N:] == 0).astype(int)
                if np.any([self.alive[-self.N:-self.N+1] == 0, np.all(self.alive[-self.N+1:]==0)]):
                    self.done = True
            
            else :
                #Results:
                #Change of state ------     
                #- from micro to MACRO
                self.state[0] = 0
                #Forager becomes active
                self.state[1] = 0
                # Food is possibly consumed.
                # Self.share = Time passed in trophallaxis
                # End of World is evalued.
                endtime = np.random.geometric(p=1-self.gamma)
                time = self.share
                self.done = ( endtime <= self.share )
                
                if not self.done:
                    # Consumption.
                    eaten = np.random.binomial(np.min([endtime, time]), self.c, size = self.N)
                    self.state[-self.N:] -= eaten
                    #Death condition is checked for all.
                    self.alive -= (self.state[-self.N:] == 0).astype(int)
                    #No Rewards
                    if np.any([self.alive[-self.N:-self.N+1] == 0, np.all(self.alive[-self.N+1:]==0)]):
                        self.done = True
        
        # Possibly Negative rewards for death.
        # rewards = -int(self.state[-self.N:] == 0)    
        return self.state, self.rewards, self.done