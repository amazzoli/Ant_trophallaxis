#ifndef NAC_H
#define NAC_H

#include "alg.h"


// Here we define the Standard Actor Critic algorithm "AC" that learns the best policy in a 
// given environment.
// The Natural Actor Critic with advantage parameters "NAC_AP" (see Bathnagar et al. 2009, algorithm 3)
// is defined as a derived class of AC where the two virtual methods for the parameter initialization
// and the actor update are overrided.
// All the methods are written in "nac.cpp"


/* Standard Actor Critic algorithm */
class MA_AC : public MARLAlgorithm {

    private:
 
        /* Trajectory of the policy parameters */ 
        vec4d policy_par_traj;
        /* Trajectory of the values */ 
        vec3d value_traj;
        /* Trajectory of the average rewards of the continuing task */ 
        vec2d av_rew_traj;
        /* Whether to save the policy trajectory */
        bool save_alg_traj;
        
        // AUX FUNCTIONS
        /* Build constant value parameters */
        vec2d const_values(double val);
        /* Build random between 0 and val */
        vec2d rand_values(double val);
        /* Build a flat policy */
        vec3d flat_policy();

    protected:  

        /* Wheather the task is continuous (gamma=1, no episode)*/
        bool continuous_task;
        
        /* Critic learning rate dependent on time */
        d_i_fnc lr_crit;
        /* Multiplicative factors on the learning rates for each player */
        vecd lr_crit_factors;
        /* Actor learning rate dependent on time */
        d_i_fnc lr_act; 
        /* Multiplicative factors on the learning rates for each player */
        vecd lr_act_factors; 
        /* Multiplicative factor on the learning rates of the average reward */
        double lr_rew_factor;

        // "CURRENT VARIABLES" CHANGED AT EACH LEARNING STEP
        /* Value/critic parameters. Index1 player, index2 state */
        vec2d curr_v_pars;
        /* Policy/actor parameters. Index1 player, index2 state, index3 action */
        vec3d curr_p_pars;
        /* Policy. Index1 player, index2 state, Index3 action */
        vec3d curr_policy;
        /* Temporal difference error */
        vecd curr_td_error;
        /* Terminal reward */
        vecd curr_t_rew;
        /* Average rewards of the continuous task */
        vecd curr_av_rew;

        // METHODS TO OVERRIDE
        virtual void init(const param& params);
        virtual void get_action(veci& action);
        virtual void learning_update(int lrn_steps_elapsed);
        virtual void build_traj();
        virtual void print_traj(str out_dir) const;

        // CHILD ALGORITHM METHODS
        virtual void child_init() {};
        virtual void critic_update();
        virtual void actor_update();

    public:

        /* Construct the algorithm given the parameters dictionary */
        MA_AC(Environment* env, const param& params, std::mt19937& generator, bool verbose=true);

        /* Algorithm description */
        const str descr() const { return "Multi-agent actor critic algorithm."; }
};


/* Natural Actor Critic with advantage parameters */
class MA_NAC_AP : public MA_AC {

    private:

        /* Advantage parameters */
        vec3d ap_par;
        vec3d grad_est;

    protected:

        void child_init();
        void actor_update();

    public:

        MA_NAC_AP(Environment* env, const param& params, std::mt19937& generator, bool verbose=true) : 
        MA_AC{env, params, generator, verbose} {};

        const str descr() const { return "Multi-agent natural actor critic with advantage parameters algorithm."; }
};


/* Actor Critic with eligibity traces */
class MA_AC_ET : public MA_AC {

    protected:

        /* ET vector actor */
        vec3d et_vec_actor;
        /* ET vector critic */
        vec2d et_vec_critic;
        /* ET factor actor */
        double lambda_actor;
        /* ET factor critic */
        double lambda_critic;

        const double et_low_bound = 0.01;
        //virtual void learning_update(int lrn_steps_elapsed);

        virtual void child_init() {};
        void critic_update();
        virtual void actor_update();

    public:
        /* Construct the algorithm given the parameters dictionary */
        MA_AC_ET(Environment* env, const param& params, std::mt19937& generator, bool verbose=true);

        /* Algorithm description */
        const str descr() const { return "Multi-agent actor critic algorithm with eligibity traces."; }
};


/* Natural Actor Critic with advantage parameters and eligibity traces */
class MA_NAC_AP_ET : public MA_AC_ET {

    private:

        /* Advantage parameters */
        vec3d ap_par;

    protected:

        void child_init();
        void actor_update();

    public:

        MA_NAC_AP_ET(Environment* env, const param& params, std::mt19937& generator, bool verbose=true) : 
        MA_AC_ET{env, params, generator, verbose} {};

        const str descr() const { return "Multi-agent natural actor critic with advantage parameters and eligibity traces algorithm."; }
};


#endif