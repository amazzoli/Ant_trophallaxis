#ifndef ENV_H
#define ENV_H

#include "utils.h"


/* Outcome of an environment transition*/
struct env_info {
    vecd reward;
    bool done;
};


/* Abstract. It contains all the info of the single player MDP to be solved
   by the Natural Actor Critic algorithm with state aggregation */
class Environment {

    protected:

        /* Random number generator */ 
        std::mt19937 generator;   

    public:

        Environment(std::mt19937& generator) : generator{generator} {};
        virtual ~Environment() {};

        /* Get the aggregate-state-space shape. Index1 player */
        virtual const int n_aggr_state(int player) const = 0;
        /* Get the number of actions. Index1 player, index2 state */
        virtual const int n_actions(int player, int state) const = 0;
        /* Get the description of each aggregate state. First index: player, second: state */
        virtual const vec2s& aggr_state_descr() const = 0;
        /* Get the description of each action. Index1 player, Index2 state, Index3 action */
        virtual const vec3s& action_descr() const = 0;

        /* Number of players */
        virtual const int n_players() const = 0;
        /* Abstract. Get the description of the environment */
        virtual const str descr() const = 0;
        /* Get the current state */
        virtual const vecd& state() = 0;
        /* Get the description of each state index */
        virtual const vecs state_descr() const = 0;

        /* Get the current aggregate state */
        virtual void aggr_state(veci& aggr_state) = 0;
        /* Set the environment in the initial state and returns the state */
        virtual void reset_state(veci& aggr_state) = 0;
        /* Environmental transition given the action which modifies the 
           internal state and return the reward and the termination flag. */
        virtual void step(const veci& action, env_info& info) = 0;
        /* Reward ot penalty in the terminal state, zero by default */
        virtual vecd terminal_reward(const double gamma, vecd& t_rew) { return vecd(n_players()); };
        /* Information about the environment */
        virtual vecd env_data() { return vecd(0); }
        virtual vecs env_data_headers() { return vecs(0); }
};



#endif