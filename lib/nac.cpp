#include "nac.h"


MA_AC::MA_AC(Environment* env, const param& params, std::mt19937& generator, bool verbose) : 
MARLAlgorithm(env, params, generator, verbose) {

    try {
        // Learning rate scheduling
        lr_crit = d_i_fnc{  // Critic
            [&params](int step) { 
                return plaw_dacay(step, params.d.at("a_burn"), params.d.at("a_expn"), params.d.at("a0"), params.d.at("ac"));
            }
        };
        lr_act = d_i_fnc{  // Actor
            [&params](int step) { 
                return plaw_dacay(step, params.d.at("b_burn"), params.d.at("b_expn"), params.d.at("b0"), params.d.at("bc"));
            }
        };
        
    } catch (std::exception) {
        throw std::invalid_argument( "Invalid learning rates in Actor Critic" );
    }
}


void MA_AC::init(const param& params){

    // Trejectory init
    policy_par_traj = vec4d(0);
    value_traj = vec3d(0);
    return_traj = vec2d(0);

    // Value parameter init
    if (params.s.find("init_val_path") != params.s.end()){
        curr_v_pars = read_vec2d( params.s.at("init_val_path") );
        if (verbose)
            std::cout << ". Value init cond from data: " << params.s.at("init_val_path") << "\n";
    }

    else{
        if (params.d.find("init_values") != params.d.end()) {
            curr_v_pars = const_values( params.d.at("init_values") );
            if (verbose)
                std::cout << ". Constant init val: " << params.d.at("init_values") << "\n";
        }
        else {
            if (params.d.find("init_values_rand") != params.d.end()){
                if (verbose)
                    std::cout << ". Random init val: " << params.d.at("init_values_rand") << "\n";
                curr_v_pars = rand_values( params.d.at("init_values_rand") );
            }
            else
                throw std::invalid_argument( "Init condition for values in AC not specified" );
        }
    }
    
    // Policy parameters init
    curr_policy = vec3d(0);
    if (params.s.find("init_pol_path") != params.s.end())
        curr_policy = read_vec3d( params.s.at("init_pol_path") );
    else
        curr_policy = flat_policy();

    curr_p_pars = vec3d(0);
    for (int p=0; p<(*env).n_players(); p++) {
        vec2d par_of_player = vec2d(0);
        for (int k=0; k<(*env).n_aggr_state(p); k++) {
            vecd par_at_state = vecd((*env).n_actions(p, k));
            pol2par_boltzmann(curr_policy[p][k], par_at_state);
            par_of_player.push_back(par_at_state);
        }
        curr_p_pars.push_back(par_of_player);
    }

    curr_td_error = vecd((*env).n_players());
    curr_t_rew = vecd((*env).n_players());

    // Child algorithm init
    child_init();
}


vec2d MA_AC::const_values(double val){
    vec2d v = vec2d((*env).n_players());
    for (int p=0; p<(*env).n_players(); p++){
        v[p] = vecd((*env).n_aggr_state(p), val);
    }
    return v;
}


vec2d MA_AC::rand_values(double val){
    std::uniform_real_distribution<double> unif_val = std::uniform_real_distribution<double>(0,val);
    vec2d v = vec2d((*env).n_players());
    for (int p=0; p<(*env).n_players(); p++){
        v[p] = vecd((*env).n_aggr_state(p), 0);
        for (int k=0; k<(*env).n_aggr_state(p); k++)
            v[p][k] = unif_val(generator);
    }
    return v;
}


vec3d MA_AC::flat_policy(){
    vec3d policy = vec3d((*env).n_players());
    for (int p=0; p<(*env).n_players(); p++){
        vec2d pol_of_player = vec2d((*env).n_aggr_state(p));
        for (int s=0; s<pol_of_player.size(); ++s) 
            pol_of_player[s] = vecd((*env).n_actions(p,s), 1.0/(double)(*env).n_actions(p,s));
        policy[p] = pol_of_player;
    }
    return policy;
}


void MA_AC::get_action(veci& action) {

    // Note: the curr_policy outside the curr_aggr_state are not computed

    for (int p=0; p<(*env).n_players(); p++) {
        par2pol_boltzmann(curr_p_pars[p][curr_aggr_state[p]], curr_policy[p][curr_aggr_state[p]]);
        std::discrete_distribution<int> dist (
            curr_policy[p][curr_aggr_state[p]].begin(), 
            curr_policy[p][curr_aggr_state[p]].end()
        );
        action[p] = dist(generator);
    }
}


void MA_AC::learning_update() {

    if (!stop_by_discount) {
        for (int p=0; p<(*env).n_players(); p++)
            curr_td_error[p] = curr_info.reward[p] + m_gamma * curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
    }
    else {
        for (int p=0; p<(*env).n_players(); p++)
            curr_td_error[p] = curr_info.reward[p] + curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
    }

    // Critic update
    curr_crit_lr = lr_crit(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++)
        curr_v_pars[p][curr_aggr_state[p]] += curr_crit_lr * curr_td_error[p];
    if (curr_info.done){
        (*env).terminal_reward(m_gamma, curr_t_rew);
        for (int p=0; p<(*env).n_players(); p++)
            curr_v_pars[p][curr_new_aggr_state[p]] += curr_crit_lr * ( curr_t_rew[p] - curr_v_pars[p][curr_new_aggr_state[p]]);
    }

    // Actor update
    curr_act_lr = lr_act(curr_step) * curr_gamma_fact;
    child_update();
}


void MA_AC::build_traj() {
    policy_par_traj.push_back(curr_p_pars);
    value_traj.push_back(curr_v_pars);
}


void MA_AC::print_traj(str out_dir) const {

    // Value trajectory
    if (traj_step > 0)
        write_vec3d(value_traj, out_dir + "value_traj.txt", (*env).aggr_state_descr());

    // Policy trajectory
    vec4d policy;
    for (int p=0; p<(*env).n_players(); p++){
        str path = out_dir + "policy" + std::to_string(p) + "_traj.txt";
        vec3d policy_of_p = vec3d(policy_par_traj.size());
        for (int t=0; t<policy_par_traj.size(); t++){
            vec2d pol_at_time = vec2d(policy_par_traj[t][p].size());
            for (int k=0; k<policy_par_traj[t][p].size(); k++){
                vecd pol_at_state = vecd(policy_par_traj[t][p][k].size());
                par2pol_boltzmann(policy_par_traj[t][p][k], pol_at_state);
                for (int a=0; a<pol_at_state.size(); a++){
                    if (pol_at_state[a]>0 && pol_at_state[a]<std::numeric_limits<double>::min())
                        pol_at_state[a] = 0;
                }
                pol_at_time[k] = pol_at_state;
            }
            policy_of_p[t] = pol_at_time;
        }
        if (traj_step > 0)
            write_vec3d(policy_of_p, path, (*env).action_descr()[p]);
        policy.push_back(policy_of_p);
    }

    // Best value and best policy
    for (int p=0; p<(*env).n_players(); p++)
        write_vec2d(policy[p][policy[p].size()-1], out_dir + "best_policy_"+std::to_string(p)+".txt");
    write_vec2d(value_traj[value_traj.size()-1], out_dir + "best_value.txt");
}


// CHILD ACTOR CRITIC

void MA_AC::child_update(){ 
    for (int p=0; p<(*env).n_players(); p++){
        for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++){
            if (a == curr_action[p]) 
                curr_p_pars[p][curr_aggr_state[p]][a] += curr_act_lr * curr_td_error[p] * (1 - curr_policy[p][curr_aggr_state[p]][a]); 
            else 
                curr_p_pars[p][curr_aggr_state[p]][a] -= curr_act_lr * curr_td_error[p] * curr_policy[p][curr_aggr_state[p]][a];
        }
    }
}



// NATURAL ACTOR CRITIC WITH ADVANTAGE PARAMETERS

void MA_NAC_AP::child_init(){
    ap_par = vec3d(0);
    for (int p=0; p<(*env).n_players(); p++) {
        vec2d ap_par_of_player = vec2d(0);
        for (int s=0; s<(*env).n_aggr_state(p); ++s) 
            ap_par_of_player.push_back(vecd((*env).n_actions(p, s)));
        ap_par.push_back(ap_par_of_player);
    }
}

void MA_NAC_AP::child_update(){ 

    for (int p=0; p<(*env).n_players(); p++) {
        double aux_t = curr_td_error[p] - ap_par[p][curr_aggr_state[p]][curr_action[p]];
        for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++)
            aux_t += curr_policy[p][curr_aggr_state[p]][a] * ap_par[p][curr_aggr_state[p]][a];
        for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++){
            if (a == curr_action[p]) 
                ap_par[p][curr_aggr_state[p]][a] += curr_crit_lr * (1 - curr_policy[p][curr_aggr_state[p]][a]) * aux_t; 
            else 
                ap_par[p][curr_aggr_state[p]][a] -= curr_crit_lr * curr_policy[p][curr_aggr_state[p]][a] * aux_t;
        }
        for (int s=0; s<ap_par[p].size(); s++)
            for (int a=0; a<curr_p_pars[p][s].size(); a++)
                curr_p_pars[p][s][a] += curr_act_lr * ap_par[p][s][a];
    }
}