#ifndef ALG_H
#define ALG_H

#include "env.h"


/* Abstract multi-agent reinforcement learning algorithm. It loops over the environment 
   and calls the algorithm-specific action for getting the action and updating 
   the learning quantities. */
class MARLAlgorithm {

    private:
        /* Uniform rand var */
        std::uniform_real_distribution<double> unif_dist;
        
    protected:

        /* MDP to solve */
        Environment* env; 
        /* Random number generator */ 
        std::mt19937 generator;
        /* Discount factor */
        double m_gamma;   
        /* If the discount factor is taken into account as stop probability */
        bool stop_by_discount;
        /* Trajectory of the returns */ 
        vec2d return_traj;
        /* Trajectory of the environmental information */
        vec2d env_info_traj;
        /* Length of the episodes */
        veci ep_len_traj;

        // "CURRENT VARIABLES" CHANGED AT EACH LEARNING STEP
        /* Aggregate state at the current time step of the learning for each player */
        veci curr_aggr_state;
        /* Chosen action at the current time step of the learning */
        veci curr_action;
        /* Current reward and terminal state info */
        env_info curr_info;
        /* Aggregate state at the current time step of the learning */
        veci curr_new_aggr_state;
        /* Current episode */
        int curr_episode;
        /* Temporal step of the episode */
        int curr_ep_step;
        /* Temporal step of the algorithm */
        int curr_step;
        /* Gamma power episode step. It remains 1 if stop_by_discount */
        float curr_gamma_fact;

        // ALGORITHM SPECIFIC METHODS
        virtual void init(const param& params) = 0;
        virtual void get_action(veci& action) = 0;
        virtual void learning_update() = 0;
        virtual void build_traj() = 0;
        virtual void print_traj(str out_dir) const = 0;

    public:

        /* Construct the algorithm given the parameters dictionary */
        MARLAlgorithm(Environment* env, const param& params, std::mt19937& generator);

        /* Algorithm description */
        virtual const str descr() const = 0;

        /* Run the algorithm */
        void run(const param& params);

        /* Print the policy, the value trajectories and their final result */
        virtual void print_output(str out_dir) const;
};


#endif