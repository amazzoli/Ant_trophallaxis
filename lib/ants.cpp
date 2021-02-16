#include "ants.h"



Ants_ma::Ants_ma(const param& par, std::mt19937& generator) : 
Environment(generator) {

	// READING PARAMETERS
	try {
		max_k = par.d.at("max_k");
		n_recipients = par.d.at("n_recipients");
		p_succ = par.d.at("p_succ");
		if (par.vecd.find("init_k") != par.vecd.end())
			init_k = par.vecd.at("init_k");
		else
			init_k = vecd( n_recipients+1, max_k+1 );
    } catch (std::exception) {
        throw std::invalid_argument( "Invalid ant-environment parameters" ); 
    }

    // INIT RANDOM GENERATORS
    unif_k_dist = std::uniform_int_distribution<int>(1, max_k);
    unif_rec_dist = std::uniform_int_distribution<int>(1, n_recipients);
    unif_dist = std::uniform_real_distribution<double>(0.0, 1.0);

    // BUILDING STATE AND ACTION INFORMATIONS
    // State
    food = veci(n_recipients+1);
    m_state = vecd(n_recipients+2);
    // Aggregate state
   	m_aggr_state_descr = vec2s( n_players(), vecs( 2*(max_k+1) ) );
	for (int p=0; p<n_players(); p++) {
		for(int k=0; k<max_k+1; k++) {
			m_aggr_state_descr[p][k] = "k=" + std::to_string(k) + "_act";
			m_aggr_state_descr[p][k+max_k+1] = "k=" + std::to_string(k) + "_wait";
		}
	}
	// Actions
	m_action_descr = vec3s( n_players(), vec2s( 2*(max_k+1), vecs(0) ) );
	for(int k=0; k<max_k+1; k++) {
		m_action_descr[0][k].push_back("gather");
		m_action_descr[0][k].push_back("share");
		m_action_descr[0][k+max_k+1].push_back("wait");
	}
	for (int p=1; p<n_players(); p++) {
		for(int k=0; k<max_k+1; k++) {
			m_action_descr[p][k].push_back("accept");
			m_action_descr[p][k].push_back("reject"); 
			m_action_descr[p][k+max_k+1].push_back("wait");
		}
	}
}


const str Ants_ma::descr() const {
	return "Ant colony with single forager and multi-recipient interactions";
}


const vecd& Ants_ma::state() {
	m_state[0] = decider;
	for (int p=0; p<n_players(); p++) 
		m_state[p+1] = food[p];
	return m_state;
}


const vecs Ants_ma::state_descr() const {
	vecs d = vecs { "decider", "forager_load" };
	for (int p=0; p<n_recipients; p++)
		d.push_back("rec" + std::to_string(p+1) + "_load");
	return d;
}


void Ants_ma::aggr_state(veci& aggr_state) {
	for (int p=0; p<n_players(); p++) {
		if (p == decider)
			aggr_state[p] = food[p];
		else
			aggr_state[p] = food[p] + max_k + 1;
	}
}


void Ants_ma::reset_state(veci& aggr_state) {

	tot_food = 0;

	// Initial food in the colony
	for (int p=0; p<n_players(); p++) {
		if (init_k[p] <= max_k && init_k[p] > 0)
			food[p] = init_k[p];
		else
			food[p] = unif_k_dist(generator);
		if (p != 0) tot_food += food[p];
	}

	// The forager starts
	decider = 0;

	Ants_ma::aggr_state(aggr_state);
}


void Ants_ma::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	for (double& r : info.reward) r = 0;

	// Forager's decision
	if (decider == 0) {
		// Gathering
		if (action[0] == 0) {
			// Success
			if (unif_dist(generator) < p_succ) food[0] = max_k;
			// Failure
			else food[0] -= 1;
		}
		// Sharing
		else {
			decider = unif_rec_dist(generator);
		}

	}
	// Recipient's decision
	else {
		// Reject or full health
		if (food[decider] == max_k || action[decider] == 1) decider = 0;
		// Accept
		else {
			food[decider] += 1;
			food[0] -= 1;
			info.reward[decider] = 1;
			info.reward[0] = 1;
			tot_food += 1;
		}
	}

	// Terminal states
	if (food[0] == 0 || tot_food == max_k * n_recipients) info.done = true;
	else info.done = false;
}



// Ants with resource consumption


Ants_consume::Ants_consume(const param& par, std::mt19937& generator) : 
Ants_ma(par, generator) {

	try {
		p_consume = par.d.at("p_consume");
		stop_at_first_death = false;
		if (par.s.find("stop_at_first_death") != par.s.end())
			if (par.s.at("stop_at_first_death") == "true" || par.s.at("stop_at_first_death") == "True")
				stop_at_first_death = true;
	} catch (std::exception) {
	    throw std::invalid_argument( "Invalid ant-environment parameters (Ants_consume model)" );
	}

	av_return = veci(n_recipients+1);
    forag_deaths_in = 0;
    forag_deaths_out = 0;
    forag_deaths_cons = 0;
    rec_deaths = veci(n_recipients);
    elapsed_steps = 0;
    forced_stops = 0;
    env_stop = true;
}


const str Ants_consume::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nUniform consumption probability";
}


void Ants_consume::reset_state(veci& aggr_state) {

	ind_rec_map = veci(n_recipients);
	for (int i=0; i<n_recipients; i++) 
		ind_rec_map[i] = i+1;
	unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);

	// A random food load for each ant 
	for (int p=0; p<n_players(); p++) {
		if (init_k[p] <= max_k && init_k[p] > 0)
			food[p] = init_k[p];
		else
			food[p] = unif_k_dist(generator);
		if (p != 0) tot_food += food[p];
	}

	// The forager starts
	decider = 0;

	Ants_ma::aggr_state(aggr_state);
}


void Ants_consume::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	for (double& r : info.reward) r = 0;
	info.done = false;
	env_stop = false;
	elapsed_steps+=lrn_steps_elapsed;

	// Forager's decision
	if (decider == 0) {
		// Gathering
		if (action[0] == 0) {
			// Success
			if (unif_dist(generator) < p_succ) food[0] = max_k;
			// Failure
			else {
				food[0] -= 1;
				// Terminal state if forager finishes food
				if (food[0] == 0) {
					forag_deaths_out++;
					info.done = true;
					env_stop = true;
				}
			} 
		}
		// Sharing
		else {
			// The new decider is a recipient with food>0, i.e. in ind_rec_map
			double u = unif_rec_dist(generator);
			decider = ind_rec_map[u];
		}
	}
	// Recipient's decision
	else {
		// Reject or full health (food==0 for the exception that the rec dies when the sharing starts)
		if (food[decider] == max_k || food[decider] == 0 || action[decider] == 1) decider = 0;
		// Accept
		else {
			food[decider] += 1;
			food[0] -= 1;
			info.reward[decider] = 1;
			info.reward[0] = 1;
			av_return[0] += 1;
			av_return[decider] += 1;
			// Terminal state if forager finishes food
			if (food[0] == 0) {
				forag_deaths_in++;
				info.done = true;
				env_stop = true;
			}
		}
	}

	// Food consumption of the recipients
	for (int p=1; p<n_recipients+1; p++) {
		if (food[p] > 0 && unif_dist(generator) < p_consume) {
			food[p] -= 1;
			// Recipient death
			if (food[p] == 0) {
				if (stop_at_first_death){
					info.done = true;
					env_stop = true;
					rec_deaths[0]++;
				}
				else {
					// Re-defining the available recipients
					auto ind_to_remove = std::remove(ind_rec_map.begin(), ind_rec_map.end(), p);
					ind_rec_map.erase(ind_to_remove);
					rec_deaths[n_recipients-ind_rec_map.size()-1]++;
					// Terminal state if all recipients are dead
					if (ind_rec_map.size() == 0) {
						info.done = true;
						env_stop = true;
					}
					else
						unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);
				}
			}
		}
	}
}


vecd Ants_consume::env_data() {

	vecd v = vecd { 
		(double)forag_deaths_in/elapsed_steps, 
		(double)forag_deaths_out/elapsed_steps,  
		(double)forag_deaths_cons/elapsed_steps,  
	};
	for (const int& d : rec_deaths) v.push_back((double)d/elapsed_steps);
	v.push_back((double)forced_stops/elapsed_steps);

	if (elapsed_steps == 0)
		for (int i=0; i<n_recipients+4; i++) v[i] = 0;

	int ep_count = forag_deaths_in+forag_deaths_out+forag_deaths_cons+forced_stops;
	if(stop_at_first_death) ep_count += rec_deaths[0];
	else ep_count += rec_deaths[rec_deaths.size()-1];

	for (int& r : av_return) {
		if (ep_count > 0){
			v.push_back(r/(float)ep_count);
			r = 0;
		}
		else v.push_back(0);
	}

	elapsed_steps = 0;
	forag_deaths_in = 0;
	forag_deaths_out = 0;
	forag_deaths_cons = 0;
	rec_deaths = veci(n_recipients);
	forced_stops = 0;

	return v;
}


vecs Ants_consume::env_data_headers() {
	vecs h = vecs { "Prob_forag_in_death\tProb_forag_out_death\tProb_forag_cons_death" };
	for (int p=0; p<n_recipients; p++)
		h.push_back("\tProb_recip_death_" + std::to_string(p));
	h.push_back("\tProb_gamma_stop");
	for (int p=0; p<n_recipients+1; p++) 
		h.push_back("\tAv_return_" + std::to_string(p));
	return h;
}


Ants_consume2::Ants_consume2(const param& par, std::mt19937& generator) : 
Ants_consume(par, generator) {

	try {
		stop_at_first_death = false;
		if (par.s.find("stop_at_first_death") != par.s.end())
			if (par.s.at("stop_at_first_death") == "true" || par.s.at("stop_at_first_death") == "True")
				stop_at_first_death = true;
	} catch (std::exception) {
	    throw std::invalid_argument( "Invalid ant-environment parameters (Ants_consume2 model)" );
	}

	gath_time_dist = std::geometric_distribution<int>(p_succ);
	cons_time_dist = std::geometric_distribution<int>(p_consume);

}


 const str Ants_consume2::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nConsumption during foraging. Fast.";	
}


void Ants_consume2::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	// By default, rewards are zero and the game continues
	for (double& r : info.reward) r = 0;
	info.done = false;
	env_stop = false;
	
	// Forager's decision
	if (decider == 0) {

		// Gathering
		if (action[0] == 0) {

			// Extracting the time for the forager to find food
			int gath_time = gath_time_dist(generator)+1;
			// Updating n steps of learning. This controls also the death by disc factor (in alg.cpp)
			lrn_steps_elapsed = gath_time;

			// Possibility of consuming food during that time
			for (int p=0; p<n_recipients+1; p++) {
				std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(gath_time, p_consume);
				consume_food(p, cons_food_dist(generator), info);
				if (info.done) {
					// Here the forager dies outside and because of gathering
					if (food[0] <= 0) forag_deaths_out++;
					break;
				}
			}

			// Gathering happens if the game doesn't stop
			if (!info.done) food[0] = max_k;  
		}	

		// Sharing
		else {
			// The new decider is a recipient with food>0, i.e. in ind_rec_map
			// No food consumed here
			double u = unif_rec_dist(generator);
			decider = ind_rec_map[u];
		}
	}

	// Recipient's decision
	else {

		// To avoid being stuck in a full-recipient full-forager loop, the food consumption of one
		// unit is imposed to the full recipient.
		// if (food[decider] >= max_k){
		// 	// Time needed for the full recipient to eat one resource
		// 	int cons_time = cons_time_dist(generator)+1;
		// 	lrn_steps_elapsed = cons_time;
		// 	consume_food(decider, 1, info);
		// 	// Imposing consuption to all the other players
		// 	std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(cons_time, p_consume);
		// 	for (int p=0; p<n_recipients+1; p++)
		// 		if (decider != p) consume_food(p, cons_food_dist(generator), info);

		// 	if (info.done) {
		// 		if (food[0] == 0) forag_deaths_cons++; // morte per consumo distinta da morte per troph
		// 	}

		// 	// After this waiting time we can imagine that the choice 
		// 	decider = 0;
		// }

		// Consuption check over all the players 
		for (int p=0; p<n_recipients+1; p++)
			if (unif_dist(generator) < p_consume) consume_food(p, 1, info);

		if (info.done) {
			if (food[0] == 0) forag_deaths_cons++; // morte per consumo dentro colonia
		}
		else {
			// Reject or full recipient
			if (food[decider] >= max_k || action[decider] == 1) decider = 0;

			// Accept
			else {
				food[decider] += 1;
				consume_food(0, 1, info);
				if (info.done)
					// morte per trophallassi
					if (food[0] == 0) forag_deaths_in++; 

				info.reward[decider] = 1;
				info.reward[0] = 1;
				av_return[0] += 1;
				av_return[decider] += 1;
			}
		}
	}

	elapsed_steps+=lrn_steps_elapsed;
}


void Ants_consume2::consume_food(int player, int amount, env_info& info){

	// Check only for a forager or a living recipient
	if (player == 0 || std::find(ind_rec_map.begin(), ind_rec_map.end(), player) != ind_rec_map.end()) {

		food[player] = std::max(0, food[player]-amount);

		// Death check
		if (food[player] <= 0) {

			// Forager death
			if (player == 0) info.done = true;

			// Recipient death
			else {

				if (stop_at_first_death) {
					rec_deaths[0]++;
					info.done = true; 
				}

				else {
					// Re-defining the available recipients
					auto ind_to_remove = std::remove(ind_rec_map.begin(), ind_rec_map.end(), player);
					ind_rec_map.erase(ind_to_remove);
					rec_deaths[n_recipients-ind_rec_map.size()-1]++;
					// Terminal state if all recipients are dead
					if (ind_rec_map.size() == 0) 
						info.done = true; 
					else
						unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);
				}
			}
		}			
		if (info.done) env_stop = true;
	}

}

// -----------------------------
// Ant environment with death penalty (explicit) and variable reward for food.
// -----------------------------


Ants_consume_death::Ants_consume_death(const param& par, std::mt19937& generator) : 
Ants_consume(par, generator) {

	try {
        stop_at_first_death = false;
		if (par.s.find("stop_at_first_death") != par.s.end())
            if (par.s.at("stop_at_first_death") == "true" || par.s.at("stop_at_first_death") == "True")
				stop_at_first_death = true;
        pen_death = par.d.at("pen_death");
        rew_eat = par.d.at("rew_eat");
        true_gamma = par.d.at("true_gamma");
        
        if (par.s.find("reward_life") != par.s.end())
            if (par.s.at("reward_life") == "true")
                rew_life = pen_death * (1 - true_gamma);
                pen_death = 0;
        
	} catch (std::exception) {
	    throw std::invalid_argument( "Invalid ant-environment parameters (Ants_consume_death model)" );
	}
    
    gath_time_dist = std::geometric_distribution<int>(p_succ);
	cons_time_dist = std::geometric_distribution<int>(p_consume);
    disc_time_dist = std::geometric_distribution<int>(1-true_gamma);
    
}


 const str Ants_consume_death::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nConsumption during foraging. Fast. Death is explicitly taken in consideration with a penalty. Stop by discount is internally calculated in environment.";	
}


void Ants_consume_death::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	// By default, rewards are zero and the game continues
	for (double& r : info.reward) r = 0;
	info.done = false;
	disc_stop = false;
	
	// Forager's decision
	if (decider == 0) {

		// Gathering
		if (action[0] == 0) {

			// Extracting the time for the forager to find food
			int gath_time = gath_time_dist(generator)+1;
            int disc_time = disc_time_dist(generator)+1;
           ; 
            if (disc_time <= gath_time) {
                info.done = true;
                disc_stop = true;
            }
            
			lrn_steps_elapsed = std::min(gath_time, disc_time);

			// Possibility of consuming food during that time
			for (int p=0; p<n_recipients+1; p++) {
                if (food[p]>0){ // skips dead ants.
                    // Consumption.
                    std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(lrn_steps_elapsed, p_consume);
                    consume_food(p, cons_food_dist(generator), info, disc_stop);
                    // Death check here and penalty.
                    if (food[p] <= 0){
                        info.reward[p] = -pen_death; 
                        av_return[p] -= pen_death;  
                    }
                    if (info.done && (!disc_stop)) {
                        // Stop because forager died outside while gathering
                        if ( (p==0) && (food[0] <= 0) ) forag_deaths_out++;
                        break;
                    }
                }
			}

			// Gathering happens if the game doesn't stop
			if (!info.done) food[0] = max_k;  
		}	

		// Sharing
		else {            
            if (unif_dist(generator) < 1-true_gamma){
                info.done = true;
                disc_stop = true;
            }
            // Consuption check over all the players anyways.
            for (int p=0; p<n_recipients+1; p++){
                if (food[p]>0){ // skips dead ants.
                    if (unif_dist(generator) < p_consume) {
                        consume_food(p, 1, info, disc_stop);
                    }
                    if (food[p]<=0) {
                        info.reward[p] = -pen_death;
                        av_return[p] -= pen_death;  
                    }
                }
            }
            if (info.done && (!disc_stop)){
                if ((food[0] == 0) && (!disc_stop)) forag_deaths_cons++; // Stop because forager died inside colony.
            }
            // The new decider is a recipient with food>0, i.e. in ind_rec_map
            // ADD STEP.
            double u = unif_rec_dist(generator);
            decider = ind_rec_map[u];
        }
	}

	// Recipient's decision
	else {

	if (unif_dist(generator) < 1-true_gamma){
            info.done = true;
            disc_stop = true;
        }
        // Consuption check over all the players anyways.
        for (int p=0; p<n_recipients+1; p++){
            if (food[p]>0){ // skips dead ants.
                if (unif_dist(generator) < p_consume) {
                    consume_food(p, 1, info, disc_stop);
                }
                if (food[p]<=0) {
                    info.reward[p] = -pen_death;
                    av_return[p] -= pen_death;  
                }
            }
        }
        if (info.done && (!disc_stop)){
            // Terminal state due to dead ants, skips the trophallaxis event.
            if ((food[0] == 0) && (!disc_stop)) forag_deaths_cons++; // Stop because forager died inside colony.
        } else {
            // Reject or full recipient
            if (food[decider] >= max_k || action[decider] == 1){
                decider = 0;
            }
            // Accept
            else {
                food[decider] += 1;
                // Recipient is rewarded.
                info.reward[decider] = rew_eat;
                av_return[decider] += rew_eat;
                
                consume_food(0, 1, info, disc_stop);
                if (info.done && (!disc_stop)) { 
                    // Death for trophallaxis.
                    info.reward[0] = -pen_death;
                    av_return[0] -= pen_death;                     
                    forag_deaths_in++; 
                } else {
                    info.reward[0] = 1;
                    av_return[0] += 1;
                }                    
            }
        }
	}

    // IF THE FORAGER DIES, I LOOK TO THE END OF THE EPISODE TO CHECK FOR THE DEATH OF THE REMAINING RECIPIENTS.
    if ((food[0] == 0) && (!disc_stop)) {
            int dead_forager_time = disc_time_dist(generator)+1;        
			lrn_steps_elapsed += dead_forager_time;

			// Possibility of consuming food during that time
			for (int p=1; p<n_recipients+1; p++) {
                if (food[p]>0){ // skips dead ants and forager.
                    // Consumption.
                    std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(dead_forager_time, p_consume);
                    consume_food(p, cons_food_dist(generator), info, disc_stop);
                    // Death check here and penalty.
                    if (food[p] <= 0){
                        info.reward[p] = -pen_death; 
                        av_return[p] -= pen_death;  
                    }
                }
			}        
    }

    for (int p=0; p<n_recipients+1; p++) {
        if (food[p]>0){
            info.reward[p] = lrn_steps_elapsed*rew_life; 
            av_return[p] += lrn_steps_elapsed*rew_life;              
        }
    }

	elapsed_steps+=lrn_steps_elapsed;
    if (disc_stop) forced_stops++;

}

// UNCHANGED
void Ants_consume_death::consume_food(int player, int amount, env_info& info, bool disc_stop){

	// Check only for a forager or a living recipient
	if (player == 0 || std::find(ind_rec_map.begin(), ind_rec_map.end(), player) != ind_rec_map.end()) { // Now superflous.

		food[player] = std::max(0, food[player]-amount);

		// Death check
		if (food[player] <= 0) {

			// Forager death
			if (player == 0) info.done = true;

			// Recipient death
			else {

				if (stop_at_first_death) {
					rec_deaths[0]++;
					info.done = true; 
				} else {
					// Re-defining the available recipients
					auto ind_to_remove = std::remove(ind_rec_map.begin(), ind_rec_map.end(), player);
					ind_rec_map.erase(ind_to_remove);
					if (!disc_stop) rec_deaths[n_recipients-ind_rec_map.size()-1]++;
					// Terminal state if all recipients are dead
					if (ind_rec_map.size() == 0) 
						info.done = true; 
					else
						unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);
				}
			}
		}
	}
}

// STRESS, NO DEATH

Ants_consume_stress::Ants_consume_stress(const param& par, std::mt19937& generator) : 
Ants_consume(par, generator) {

	try {
        stop_at_first_death = false;
        pen_stress = par.d.at("pen_stress");
        rew_eat = par.d.at("rew_eat");
        
        rew_life = 0;
        if (par.s.find("reward_life") != par.s.end()) {
            if (par.s.at("reward_life") == "true"){
                rew_life = pen_stress;
                pen_stress = 0;
            }
        }
        
	} catch (std::exception) {
	    throw std::invalid_argument( "Invalid ant-environment parameters (Ants_consume_stress model)" );
	}
}


 const str Ants_consume_stress::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nConsumption during foraging. Fast. Having an empty crop is explicitly taken in consideration with a penalty. Suited for continuous task.";	
}


void Ants_consume_stress::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	// By default, rewards are zero and the game continues
	for (double& r : info.reward) r = 0;
	info.done = false;
	
	// Forager's decision
	if (decider == 0) {

		// Gathering
		if (action[0] == 0) {

			// Possibility of consuming food during that time
			for (int p=0; p<n_recipients+1; p++) {
                if (unif_dist(generator) < p_consume) {
                    // Consumption.
                    consume_food(p, 1, info);
                    if ( (p==0) && (food[0] <= 0) ) forag_deaths_out++; // Forager stressed outside colony.            
                }
            }
            
			// Gathering happens
            double p_succ_eff = p_succ;
            // If forager is stressed, p_succ is decreased
            if (food[0] == 0){
                p_succ_eff = std::min( p_succ, p_consume/max_k);
            }
            if (unif_dist(generator) < p_succ_eff) {
                food[0] = max_k;
            }
		}	

		// Sharing
		else {            
            // Consuption check over all the players anyways.
            for (int p=0; p<n_recipients+1; p++){
                if (unif_dist(generator) < p_consume) {
                    consume_food(p, 1, info);
                }
            }
            if (food[0] == 0) forag_deaths_cons++; // Forager stressed inside colony.

            // The new decider is any recipient, i.e. in ind_rec_map
            double u = unif_rec_dist(generator);
            decider = ind_rec_map[u];
        }
	} else {
	// Recipient's decision

        // Consuption check over all the players anyways.
        for (int p=0; p<n_recipients+1; p++){
            if (unif_dist(generator) < p_consume) {
                consume_food(p, 1, info);
            }
        }

        if (food[0] == 0){
           forag_deaths_cons++; // Stop because forager died inside colony.
           decider = 0;
        }
        // Reject or full recipient
        else if (food[decider] >= max_k || action[decider] == 1){
            decider = 0;
        }  
        else {
        // Accept
            food[decider] += 1;
            // Recipient is possibly rewarded.
            info.reward[decider] = rew_eat;
            av_return[decider] += rew_eat;
            
            consume_food(0, 1, info);
            if ((food[0] == 0)) {
                forag_deaths_in++; // Forager died inside colony.
                decider = 0;
            }
            info.reward[0] = 1;
            av_return[0] += 1;
        }                    
    }

    int stressed;
    stressed = 0;
    
    for (int p=0; p<n_recipients+1; p++) {
        if (food[p]>0){
            info.reward[p] += rew_life; 
            av_return[p] += rew_life;              
        } else {
            if (p>0) stressed++;
            info.reward[p] -= pen_stress; 
            av_return[p] -= pen_stress;                      
        }
    }
    
    if (stressed > 0)
        rec_deaths[stressed-1]++ ;

}

// Basically useless now.
void Ants_consume_stress::consume_food(int player, int amount, env_info& info){
    food[player] = std::max(0, food[player]-amount);
}

