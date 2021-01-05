#ifndef ANTS_H
#define ANTS_H


#include "env.h"


class Ants_ma : public Environment {

    protected:

        // PARAMETERS
        // Maximum food load of each ant
        int max_k;
        // Number of recipent ants
        int n_recipients;
        // Foraging success probability
        float p_succ;
        // Initial food in the colony. If elem>max_k the food is random btw 1 and max_k. Random by default
        vecd init_k;

        // STATE VARIABLES
        // Amount of food in each of the n_recipients+1 ants
        veci food;
        // Which ant is the decider
        int decider;
        // Total food in the colony
        float tot_food;

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

        const int n_players() const { return n_recipients+1; }
        virtual const str descr() const; 
	    const vecd& state();
	    const vecs state_descr() const;

        void aggr_state(veci& aggr_state);
        virtual void reset_state(veci& aggr_state);
        virtual void step(const veci& action, env_info& info);
};


class Ants_consume : public Ants_ma {

    private:

        // PARAMETERS
        // Probability that an ant consumes one unit of food each step
        double p_consume;

        // List of recipient indexes that are not death
        veci ind_rec_map;

        // INFO TRAJECTORY
        /* Average return */
        veci av_return;
        /* Episodes ended by forager's death */
        int forag_deaths;
        /* Episodes ended by all recipients death */
        int rec_deaths;
        /* Episodes ended discount forced stop */
        int forced_stops;
        /* Aux var to control the forced stop */
        bool env_stop;
        /* N. steps from the last writing of the trajectory quantities */
        double elapsed_steps;

    public:

        Ants_consume(const param& par, std::mt19937& generator);
        const str descr() const; 
        void reset_state(veci& aggr_state);
        void step(const veci& action, env_info& info);

        vecd env_data();
        vecs env_data_headers();
};


#endif