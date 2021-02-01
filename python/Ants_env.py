import numpy as np

class AntsEnv():

    def __init__(self, Nr=1, Mmax=10, c=0.01, rg=0.05, gamma=0.99, ran_init = False):
        # contains initialization data
        self.done = False
        # initial state initialization
        self.Nr = Nr
        self.N = 1 + self.Nr
        self.Mmax = Mmax
        # state is M/m (MACRO/micro state), ID_active, c_forager, c_recipient1, c_recipient2, ...   
        self.state = np.zeros(3+Nr, dtype='int')
        self.state[:2] = 0
        if ran_init:
            self.state[2] = np.random.randint(1,11)
            self.state[3:] = np.random.randint(1,11,self.N)
        else :
            self.state[2] = self.Mmax//2
            self.state[3:] = self.Mmax//2            
            
        self.alive = np.ones(1+Nr, dtype='bool')
        self.c = c
        self.rg = rg
        self.gamma = gamma
        # length of last step of sharing
        self.share = 1
        self.anyforager = True
        
    def get_state(self):
        return self.state
    
    def step(self, action):
        # needs to
        # apply change to internal state
        # calculate rewards
        # returns observation, rewards, done_flag
        self.rewards = np.zeros(self.N)
        time = 0 # passing time
        
        
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
                time = 0

            # Gather(0)
            else :
                # Results:
                # Change of state ------
                # Evaluation gathering time for success.
                time = np.random.geometric(p=self.rg)
                # End of World is evalued.
                endtime = np.random.geometric(p=1-self.gamma)
                self.done = ( endtime <= time )
                
                if self.done:
                    time = np.min([endtime, time])
                    
                # Consumption.
                eaten = np.random.binomial(time, self.c, size=self.N)
                self.state[-self.N:] -= eaten

                # Death condition is checked for all.
                # Penalty for death!

                self.alive = (self.state[-self.N:] > 0)

                # All foragers dead.
                if np.all(self.alive[-self.N:-self.N+1] == 0):
                    self.done = True
                    # CHECK IF RECEIVERS DIE BEFORE END OF WORLD.
                    endtime = np.random.geometric(p=1-self.gamma)
                    time += endtime
                    endfood = np.random.binomial(endtime, self.c, size=self.N)
                    self.state[-self.N:] -= endfood
                    self.alive = (self.state[-self.N:] > 0)

                if np.all(self.alive[-self.N+1:] == 0):
                    self.done = True

                self.rewards -= 10 * (np.logical_not(self.alive))
                if not self.done:
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
                action = 0
            
            if bool(action):
                # Take(1)
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
                # Penalty for death!
                self.alive = (self.state[-self.N:] > 0)
                
                if np.all(self.alive[-self.N:-self.N+1] == 0):
                    self.done = True
                    time = self.share
                    # CHECK IF RECEIVERS DIE BEFORE END OF WORLD.
                    endtime = np.random.geometric(p=1-self.gamma)
                    time += endtime
                    endfood = np.random.binomial(endtime, self.c, size=self.N)
                    self.state[-self.N:] -= endfood
                    self.alive = (self.state[-self.N:] > 0)


                if np.all(self.alive[-self.N+1:] == 0):
                    self.done = True
                    if np.all(self.alive[-self.N:-self.N+1] > 0):
                        time = self.share
                
                # Penalty for death!
                self.rewards -= 10 * (np.logical_not(self.alive))

            
            else :
                # Pass(0) 
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
                    
                    # Death condition is checked for all.
                    # Penalty for death!
                    self.alive = (self.state[-self.N:] > 0)

                    #No Rewards - Only Death
                    if np.all(self.alive[-self.N:-self.N+1] == 0):
                        self.done = True
                        time = self.share
                        # CHECK IF RECEIVERS DIE BEFORE END OF WORLD.
                        endtime = np.random.geometric(p=1-self.gamma)
                        time += endtime
                        endfood = np.random.binomial(endtime, self.c, size=self.N)
                        self.state[-self.N:] -= endfood
                        self.alive = (self.state[-self.N:] > 0)

                    if np.all(self.alive[-self.N+1:] == 0):
                        self.done = True
                        time = self.share

                    self.rewards -= 10 * (np.logical_not(self.alive))

        
        # Possibly Negative rewards for death.
        # rewards = -int(self.state[-self.N:] == 0)    
        return self.state, self.rewards, self.done, time