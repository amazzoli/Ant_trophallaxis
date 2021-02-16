#ifndef ANTS_H
#define ANTS_H


#include "env.h"


// Forager - Recipients model. The forager gathers with p_succ probaility and it fills 
// its load to max_k. The failure leads to a loss of 1 res. The sharing action leads to
// the choice of a randomly chosen recipient which accept 1 res until
// it rejects and gives back the choice to the forager.
class Ants_ma : public Environment {

    protected:

        // PARAMETERS
        // Maximum food load of each ant
        int max_k;
        // Number of recipent ants
        int n_recipients;
        // Foraging success probability
        double p_succ;
        // Initial food in the colony. If elem>max_k the food is random btw 1 and max_k. Random by default
        vecd init_k;

        // STATE VARIABLES
        // Amount of food in each of the n_recipients+1 ants
        veci food;
        // Which ant is the decider
        int decider;
        // Total food in the colony
        double tot_food;

        // AUX VARIABLES
        /* It contains all the information about environment */     
        vecd m_state;
        /* Description of aggregate states. Index1 player, index2 aggr_state */     
        vec2s m_aggr_state_descr;
        /* Description of actions. Index1 player, index2 aggr_state, index3 action */     
        vec3s m_action_descr;
        /* Uniform distribution over the food load */
        std::uniform_int_distribution<int> unif_k_dist;
        /* Uniform distribution over the recipients */
        std::uniform_int_distribution<int> unif_rec_dist;
        /* Uniform rand var */
        std::uniform_real_distribution<double> unif_dist;

        vec2s set_aggr_state_descr();
        vec3s set_action_descr();

    public:

        Ants_ma(const param& par, std::mt19937& generator);

        const int n_aggr_state(int player) const { return m_aggr_state_descr[player].size(); }
        const int n_actions(int player, int state) const { return m_action_descr[player][state].size(); }
        const vec2s& aggr_state_descr() const { return m_aggr_state_descr; }
        const vec3s& action_descr() const { return m_action_descr; }
        virtual void gamma_stop(const int& lrn_steps_elapsed) {};

        const int n_players() const { return n_recipients+1; }
        virtual const str descr() const; 
	    const vecd& state();
	    const vecs state_descr() const;

        void aggr_state(veci& aggr_state);
        virtual void reset_state(veci& aggr_state);
        virtual void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
};


// Forager - recipients model having resource consuption. The recipients can consume
// 1 res at each step.
class Ants_consume : public Ants_ma {

    protected:

        // PARAMETERS
        // Probability that an ant consumes one unit of food each step
        double p_consume;
        // Whether the game stops at the first death recipient
        bool stop_at_first_death;

        // List of recipient indexes that are not death
        veci ind_rec_map;

        // INFO TRAJECTORY
        /* Average return */
        veci av_return;
        /* Episodes ended by forager's death inside colony */
        int forag_deaths_in;
        /* Episodes ended by forager's death outside colony */
        int forag_deaths_out;
        /* Episodes ended by forager's death outside colony */
        int forag_deaths_cons;
        /* Episodes ended by all recipients death */
        veci rec_deaths;
        /* Episodes ended discount forced stop */
        int forced_stops;
        /* Aux var to control the forced stop */
        bool env_stop;
        /* N. steps from the last writing of the trajectory quantities */
        double elapsed_steps;

    public:

        Ants_consume(const param& par, std::mt19937& generator);
        virtual const str descr() const; 
        void reset_state(veci& aggr_state);
        virtual void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
        void gamma_stop(const int& lrn_steps_diff) {forced_stops++; elapsed_steps-=lrn_steps_diff; }
        vecd env_data();
        vecs env_data_headers();
};


// Forager - recipients model having resource consuption. The recipients can consume
// 1 res at each step only when the decision is on the forager side. The forager
// does not loose res by failed gahterings.
// class Ants_consume2 : public Ants_consume {

//     public:
//         Ants_consume2(const param& par, std::mt19937& generator) : Ants_consume(par, generator) {};
//         const str descr() const; 
//         void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
// };



class Ants_consume2 : public Ants_consume {

    private:
        std::geometric_distribution<int> gath_time_dist;
        std::geometric_distribution<int> cons_time_dist;
        void consume_food(int player, int amount, env_info& info);

    public:
        Ants_consume2(const param& par, std::mt19937& generator);
        const str descr() const; 
        void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
};

class Ants_consume_death : public Ants_consume {

    protected: 
        double pen_death;
        double rew_eat;
        double true_gamma;
        double rew_life = 0;
        /* Aux var to control the stop by discount*/
        bool disc_stop;

    private:
        std::geometric_distribution<int> gath_time_dist;
        std::geometric_distribution<int> cons_time_dist;
        std::geometric_distribution<int> disc_time_dist;
        void consume_food(int player, int amount, env_info& info, bool disc_stop);

    public:
        Ants_consume_death(const param& par, std::mt19937& generator);
        const str descr() const; 
        void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
};


class Ants_consume_stress : public Ants_consume {

    protected: 
        double pen_stress;
        double rew_eat;
        double rew_life;


    private:
        void consume_food(int player, int amount, env_info& info);

    public:
        Ants_consume_stress(const param& par, std::mt19937& generator);
        const str descr() const; 
        void step(const veci& action, env_info& info, int& lrn_steps_elapsed);
        vecd env_data();
};


#endif