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


void Ants_ma::step(const veci& action, env_info& info) {

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

    forag_deaths = 0;
    rec_deaths = 0;
    enlapsed_steps = 0;
    forced_stops = 0;
    env_stop = true;
}


const str Ants_consume::descr() const {
	return "Ant colony with single forager and multi-recipient interactions.\nUniform consumption probability";
}


void Ants_consume::reset_state(veci& aggr_state) {

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


void Ants_consume::step(const veci& action, env_info& info) {

	for (double& r : info.reward) r = 0;
	info.done = false;
	env_stop = false;
	enlapsed_steps++;

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
					forag_deaths++;
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
			// Terminal state if forager finishes food
			if (food[0] == 0) {
				forag_deaths++;
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
	vecd v = vecd { (double)forag_deaths/enlapsed_steps, (double)rec_deaths/enlapsed_steps, (double)forced_stops/enlapsed_steps };
	if (enlapsed_steps == 0){
		v[0] = 0; v[1] = 0;
	}
	enlapsed_steps = 0;
	forag_deaths = 0;
	rec_deaths = 0;
	forced_stops = 0;
	return v;
}


vecs Ants_consume::env_data_headers() {
	return vecs { "Prob_forager_deaths\tProb_recipients_deaths" };
}