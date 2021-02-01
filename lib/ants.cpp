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
	} catch (std::exception) {
	    throw std::invalid_argument( "Invalid ant-environment parameters (Ants_consume model)" );
	}

	av_return = veci(n_recipients+1);
    forag_deaths_in = 0;
    forag_deaths_out = 0;
    rec_deaths = 0;
    elapsed_steps = 0;
    forced_stops = 0;
    env_stop = true;
}


const str Ants_consume::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nUniform consumption probability";
}


void Ants_consume::reset_state(veci& aggr_state) {

	//std::cout << "Reset state\n";

	if (!env_stop) forced_stops++;

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
	elapsed_steps++;

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
				// Re-defining the available recipients
				auto ind_to_remove = std::remove(ind_rec_map.begin(), ind_rec_map.end(), p);
				ind_rec_map.erase(ind_to_remove);
				// Terminal state if all recipients are dead
				if (ind_rec_map.size() == 0) {
					rec_deaths++;
					info.done = true;
					env_stop = true;
				}
				else
					unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);
			}
		}
	}
}


vecd Ants_consume::env_data() {

	vecd v = vecd { 
		(double)forag_deaths_in/elapsed_steps, 
		(double)forag_deaths_out/elapsed_steps, 
		(double)rec_deaths/elapsed_steps, 
		(double)forced_stops/elapsed_steps 
	};
	if (elapsed_steps == 0){
		v[0] = 0; v[1] = 0; v[2] = 0; v[3] = 0;
	}

	int ep_count = forag_deaths_in+forag_deaths_out+rec_deaths+forced_stops;
	for (int& r : av_return) {
		if (ep_count > 0) v.push_back(r/(float)ep_count);
		else v.push_back(0);
		r = 0;
	}

	elapsed_steps = 0;
	forag_deaths_in = 0;
	forag_deaths_out = 0;
	rec_deaths = 0;
	forced_stops = 0;

	return v;
}


vecs Ants_consume::env_data_headers() {
	vecs h = vecs { "Prob_forag_in_death\tProb_forag_out_death\tProb_recipients_death\tProb_gamma_stop" };
	for (int p=0; p<n_recipients+1; p++) {
		h.push_back("\tAv_return_" + std::to_string(p));
	}
	return h;
}


Ants_consume2::Ants_consume2(const param& par, std::mt19937& generator) : 
Ants_consume(par, generator) {
	gath_time_dist = std::geometric_distribution<int>(p_succ);
	cons_time_dist = std::geometric_distribution<int>(p_consume);
}


 const str Ants_consume2::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nConsumption during foraging. Fast.";	
}


void Ants_consume2::step(const veci& action, env_info& info, int& lrn_steps_elapsed) {

	for (double& r : info.reward) r = 0;
	info.done = false;
	env_stop = false;
	
	// Forager's decision
	if (decider == 0) {

		// Gathering
		if (action[0] == 0) {

			// Extracting the time for the forager to find food
			int gath_time = gath_time_dist(generator)+1;
			lrn_steps_elapsed = gath_time;

			// Possibility of consuming food during that time
			// std::cout << ". new food:  ";
			for (int p=0; p<n_recipients+1; p++) {
				std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(gath_time, p_consume);
				consume_food(p, cons_food_dist(generator), info);
				if (info.done) {
					if (food[0] == 0) forag_deaths_out++;
					else rec_deaths++;
					break;
				}
			}

			// Gathering happens if forager did't die before
			if (food[0] > 0) food[0] = max_k;  
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
		// To avoid being stuck in a full-recipient full-forager loop, the food consumption of one
		// unit is imposed to the full recipient.
		if (food[decider] >= max_k){
			int cons_time = cons_time_dist(generator)+1;
			lrn_steps_elapsed = cons_time;
			consume_food(decider, 1, info);
			std::binomial_distribution<int> cons_food_dist = std::binomial_distribution<int>(cons_time, p_consume);
			for (int p=0; p<n_recipients+1; p++) {
				if (decider != p) {
					consume_food(p, cons_food_dist(generator), info);
				}
			}

			if (info.done) {
				if (food[0] == 0) forag_deaths_out++; // morte per consumo distinta da morte per troph
				else rec_deaths++;
			}

			decider = 0;
		}
		// Otherwise there is a consuption check over all the players and the recipient acts
		else {
			for (int p=0; p<n_recipients+1; p++)
				if (unif_dist(generator) < p_consume) consume_food(p, 1, info);

			if (info.done) {
				if (food[0] == 0) forag_deaths_out++; // morte per consumo distinta da morte per troph
				else rec_deaths++;
			}
			// Reject
			if (action[decider] == 1) {
				decider = 0;
			}
			// Accept
			else {
				food[decider] += 1;
				consume_food(0, 1, info);
				if (info.done) {
					if (food[0] == 0) forag_deaths_in++; // morte per consumo distinta da morte per troph
					else rec_deaths++;
				}
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

	food[player] = std::max(0, food[player]-amount);

	// Death check
	if (food[player] <= 0) {

		// Forager death
		if (player == 0)  info.done = true;

		// Recipient death
		else{
			// Re-defining the available recipients
			auto ind_to_remove = std::remove(ind_rec_map.begin(), ind_rec_map.end(), player);
			ind_rec_map.erase(ind_to_remove);
			// Terminal state if all recipients are dead
			if (ind_rec_map.size() == 0) 
				info.done = true; 
			else
				unif_rec_dist = std::uniform_int_distribution<int>(0, ind_rec_map.size()-1);
		}
	}			
	if (info.done) env_stop = true;
}