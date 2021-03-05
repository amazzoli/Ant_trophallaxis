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

        //Learning rate factors
        if (params.vecd.find("a_lr_factor") != params.vecd.end())
            lr_act_factors = params.vecd.at("a_lr_factor");
        else
            lr_act_factors = vecd((*env).n_players(), 1);
        if (params.vecd.find("c_lr_factor") != params.vecd.end())
            lr_crit_factors = params.vecd.at("c_lr_factor");
        else
            lr_crit_factors = vecd((*env).n_players(), 1);

        // Continuous task conditions
        continuous_task = false;
        if (params.s.find("continuous_task") != params.s.end()) {
            if (params.s.at("continuous_task") == "true" || params.s.at("continuous_task") == "True") {
                continuous_task = true;
                lr_rew_factor = params.d.at("lr_rew_factor");
                m_gamma = 1;
                stop_by_discount = false;
            }
        }
        
        // Simplex clipping (only for NAC and NAC_ET)
        simpl_clip = false;
        simpl_clip_eps = 0.0;
        if (params.s.find("simplex_clipping") != params.s.end()) {
            if (params.s.at("simplex_clipping") == "true" || params.s.at("simplex_clipping") == "True") {
                simpl_clip = true;
                simpl_clip_eps = params.d.at("simplex_clipping_epsilon");
            }
        }
        
        
        

    } catch (std::exception) {
        throw std::invalid_argument( "Invalid learning rates in Actor Critic" );
    }
}


void MA_AC::init(const param& params){

    // Trajectory init
    save_alg_traj = true;
    if (params.s.find("save_alg_traj") != params.s.end()) 
        if (params.s.at("save_alg_traj") == "false" || params.s.at("save_alg_traj") == "False")
            save_alg_traj = false;
    policy_par_traj = vec4d(0);
    value_traj = vec3d(0);
    av_rew_traj = vec2d(0);

    // Value parameter init
    if (params.s.find("init_val_path") != params.s.end()){
        curr_v_pars = read_vec2d( params.s.at("init_val_path") );
        if (verbose)
            std::cout << "Value init cond from data: " << params.s.at("init_val_path") << "\n";
    }
    else{
        if (params.d.find("init_values") != params.d.end()) {
            curr_v_pars = const_values( params.d.at("init_values") );
            if (verbose)
                std::cout << "Constant init val: " << params.d.at("init_values") << "\n";
        }
        else {
            if (params.d.find("init_values_rand") != params.d.end()){
                if (verbose)
                    std::cout << "Random init val: " << params.d.at("init_values_rand") << "\n";
                curr_v_pars = rand_values( params.d.at("init_values_rand") );
            }
            else
                throw std::invalid_argument( "Init condition for values in AC not specified" );
        }
    }
    
    // Policy init
    curr_policy = vec3d((*env).n_players());
    if (params.s.find("init_pol_dir") != params.s.end()){
        std::cout << "Init policy from data: " << params.s.at("init_pol_dir") << "\n";
        for (int p=0; p<(*env).n_players(); p++) {
            curr_policy[p] = read_vec2d( params.s.at("init_pol_dir") + "init_pol" + std::to_string(p) + ".txt", true ); 
        }
    }
    else
        curr_policy = flat_policy();

    // Policy parameters init
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
    curr_av_rew =  vecd((*env).n_players());

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

    // std::cout << " pol";
    for (int p=0; p<(*env).n_players(); p++) {
        // std::cout << p << ": ";
        par2pol_boltzmann(curr_p_pars[p][curr_aggr_state[p]], curr_policy[p][curr_aggr_state[p]]);
        std::discrete_distribution<int> dist (
            curr_policy[p][curr_aggr_state[p]].begin(), 
            curr_policy[p][curr_aggr_state[p]].end()
        );
        // for (double& pol : curr_policy[p][curr_aggr_state[p]])
        //     std::cout << pol << " ";
        action[p] = dist(generator);
    }
}


void MA_AC::learning_update(int lrn_steps_elapsed) {

    double eff_gamma = 1;
    if (!stop_by_discount) eff_gamma = m_gamma; 
    
    if (continuous_task){
        if (lrn_steps_elapsed > 1)
            std::cout << "Warning: learning step > 1 in continuous task\n";
        if (curr_info.done)
            std::cout << "Warning: episode done at step " << curr_step << " in continuous task\n";
    }

    for (int t=0; t<lrn_steps_elapsed; t++) {

        // TD is given by a self loop with zero reward
        if (t < lrn_steps_elapsed-1) 
            for (int p=0; p<(*env).n_players(); p++) 
                curr_td_error[p] = eff_gamma * curr_v_pars[p][curr_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
        // normal TD for the last step
        else {
            if (curr_info.done) {
                //(*env).terminal_reward(eff_gamma, curr_t_rew);
                for (int p=0; p<(*env).n_players(); p++)
                    //curr_td_error[p] = curr_t_rew[p] - curr_v_pars[p][curr_aggr_state[p]];
                    curr_td_error[p] = curr_info.reward[p] - curr_v_pars[p][curr_aggr_state[p]];
            }
            else {

                for (int p=0; p<(*env).n_players(); p++){
                    if (continuous_task) {
                        double lrr = lr_crit(curr_step) * lr_rew_factor;
                        curr_av_rew[p] = (1 - lrr) * curr_av_rew[p] + lrr * curr_info.reward[p];
                        curr_td_error[p] = curr_info.reward[p] - curr_av_rew[p] + curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
                    }
                    else
                        curr_td_error[p] = curr_info.reward[p] + eff_gamma * curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
                }
            }
        }

        critic_update();
        actor_update();
    }
}


void MA_AC::actor_update() { 
    // Actor update (curr_gamma_fact is 1 if stop_by_discount)
    double aux_lr = lr_act(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++){
        double lr = aux_lr * lr_act_factors[p];
        for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++){
            if (a == curr_action[p]) 
                curr_p_pars[p][curr_aggr_state[p]][a] += lr * curr_td_error[p] * (1 - curr_policy[p][curr_aggr_state[p]][a]); 
            else 
                curr_p_pars[p][curr_aggr_state[p]][a] -= lr * curr_td_error[p] * curr_policy[p][curr_aggr_state[p]][a];
        }
    }
}


void MA_AC::critic_update() {
    // Critic update (curr_gamma_fact is 1 if stop_by_discount)
    double aux_lr = lr_crit(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++) {
        curr_v_pars[p][curr_aggr_state[p]] += aux_lr * lr_crit_factors[p] * curr_td_error[p];
    }
}


void MA_AC::build_traj() {

    policy_par_traj.push_back(curr_p_pars);
    value_traj.push_back(curr_v_pars);
    av_rew_traj.push_back(curr_av_rew);
}


void MA_AC::print_traj(str out_dir) const {

    // Value trajectory
    if (save_alg_traj && traj_step > 0)
        write_vec3d(value_traj, out_dir + "value_traj.txt", (*env).aggr_state_descr());

    // Value trajectory
    if (save_alg_traj && traj_step > 0 && continuous_task)
        write_vec2d(av_rew_traj, out_dir + "av_reward_traj.txt");

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
        if (save_alg_traj && traj_step > 0)
            write_vec3d(policy_of_p, path, (*env).action_descr()[p]);
        policy.push_back(policy_of_p);
    }

    // Best value and best policy
    for (int p=0; p<(*env).n_players(); p++)
        write_vec2d(policy[p][policy[p].size()-1], out_dir + "best_policy_"+std::to_string(p)+".txt");
    write_vec2d(value_traj[value_traj.size()-1], out_dir + "best_value.txt");
}


// NATURAL ACTOR CRITIC WITH ADVANTAGE PARAMETERS

void MA_NAC_AP::child_init(){
    ap_par = vec3d(0);
    grad_est = vec3d(0);
    for (int p=0; p<(*env).n_players(); p++) {
        vec2d ap_par_of_player = vec2d(0);
        vec2d grad_est_of_player = vec2d(0);
        for (int s=0; s<(*env).n_aggr_state(p); ++s) {
            ap_par_of_player.push_back(vecd((*env).n_actions(p, s)));
            grad_est_of_player.push_back(vecd((*env).n_actions(p, s)));
        }
        ap_par.push_back(ap_par_of_player);
        grad_est.push_back(grad_est_of_player);
    }
}

void MA_NAC_AP::actor_update(){ 

    double aux_crit_lr = lr_crit(curr_step) * curr_gamma_fact;
    double aux_act_lr = lr_act(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++) {
        double lr_crit = aux_crit_lr * lr_crit_factors[p];
        double lr_act = aux_act_lr * lr_act_factors[p];
        for (int s=0; s<ap_par[p].size(); s++) {
            for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++) {
                if (s == curr_aggr_state[p]){
                    if (a == curr_action[p])
                        grad_est[p][s][a] = 1 - curr_policy[p][s][a];
                    else
                        grad_est[p][s][a] = - curr_policy[p][s][a];
                }
                else
                    grad_est[p][s][a] = 0;
            }

            double aux_t = curr_td_error[p];
            for (int a=0; a<curr_p_pars[p][s].size(); a++)
                aux_t -= grad_est[p][s][a] * ap_par[p][s][a];

            for (int a=0; a<curr_p_pars[p][s].size(); a++){
                ap_par[p][s][a] += lr_crit * grad_est[p][s][a] * aux_t;
                curr_p_pars[p][s][a] += lr_act * ap_par[p][s][a];
            }
            
            // Simplex Clipping
            int n_act = curr_p_pars[p][s].size();
            if (simpl_clip && (n_act > 1) ){
                vecd temp_policy = vecd(n_act);
                par2pol_boltzmann(curr_p_pars[p][s], temp_policy);
                // WORKING ONLY IF N_ACTIONS = 2
                for (int a=0; a < n_act; a++){
                    curr_policy[p][s][a] = std::max( std::min(temp_policy[a], (1-simpl_clip_eps*(n_act-1))), simpl_clip_eps);
                }
                pol2par_boltzmann(curr_policy[p][s], curr_p_pars[p][s]);
            }
        }
    }
}



MA_AC_ET::MA_AC_ET(Environment* env, const param& params, std::mt19937& generator, bool verbose) : 
MA_AC(env, params, generator, verbose) {

    try {
        lambda_actor = params.d.at("lambda_actor");
        lambda_critic = params.d.at("lambda_critic");       
    } catch (std::exception) {
        throw std::invalid_argument( "Invalid learning rates in Actor Critic ET" );
    }

    et_vec_actor = vec3d((*env).n_players());
    for (int p=0; p<(*env).n_players(); p++){
        vec2d vec_of_player = vec2d((*env).n_aggr_state(p));
        for (int s=0; s<vec_of_player.size(); ++s) 
            vec_of_player[s] = vecd((*env).n_actions(p,s));
        et_vec_actor[p] = vec_of_player;
    }
    et_vec_critic = vec2d((*env).n_players(), vecd());
    for (int p=0; p<(*env).n_players(); p++){
        et_vec_critic[p] = vecd((*env).n_aggr_state(p), 0);
    }
}


// void MA_AC_ET::learning_update(int lrn_steps_elapsed) {

//     // WARNING: !stop_by_discount NEVER CHECKED!!

//     double eff_gamma = 1;
//     if (!stop_by_discount) eff_gamma = m_gamma; 

//     if (continuous_task){
//         if (lrn_steps_elapsed > 1)
//             std::cout << "Warning: learning step > 1 in continuous task\n";
//         if (curr_info.done)
//             std::cout << "Warning: episode done at step " << curr_step << " in continuous task\n";
//     }

//     for (int t = 0; t<lrn_steps_elapsed; t++) {

//         // TD is given by a self loop with zero reward
//         if (t < lrn_steps_elapsed-1) 
//             for (int p=0; p<(*env).n_players(); p++) 
//                 curr_td_error[p] = eff_gamma * curr_v_pars[p][curr_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
//         // normal TD for the last step
//         else {
//             if (curr_info.done) {
//                 //(*env).terminal_reward(eff_gamma, curr_t_rew);
//                 for (int p=0; p<(*env).n_players(); p++)
//                     //curr_td_error[p] = curr_t_rew[p] - curr_v_pars[p][curr_aggr_state[p]];
//                     curr_td_error[p] = curr_info.reward[p] - curr_v_pars[p][curr_aggr_state[p]];
//             }
//             else {
//                 for (int p=0; p<(*env).n_players(); p++){
//                     if (continuous_task) {
//                         double lrr = lr_crit(curr_step) * lr_rew_factor;
//                         curr_av_rew[p] = (1 - lrr) * curr_av_rew[p] + lrr * curr_info.reward[p];
//                         curr_td_error[p] = curr_info.reward[p] - curr_av_rew[p] + curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
//                     }
//                     else
//                         curr_td_error[p] = curr_info.reward[p] + eff_gamma * curr_v_pars[p][curr_new_aggr_state[p]] - curr_v_pars[p][curr_aggr_state[p]];
//                 }
//             }
//         }

//         critic_update();
//         // Actor update (curr_gamma_fact is 1 if stop_by_discount)
//         actor_update();
//     }

//     // If terminal state, the eligibity vectors are re-init to zero
//     if (curr_info.done) {
//         for (int p=0; p<(*env).n_players(); p++) {
//             for (int s=0; s<(*env).n_aggr_state(p); ++s) {
//                 et_vec_critic[p][s] = 0;
//                 for (int a=0; a<(*env).n_actions(p,s); ++a)
//                     et_vec_actor[p][s][a] = 0;
//             }
//         }
//     }
// }



void MA_AC_ET::critic_update() {
    // Critic update (curr_gamma_fact is 1 if stop_by_discount)
    double aux_lr = lr_crit(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++){
        double lr = aux_lr * lr_crit_factors[p];
        for (int s=0; s<(*env).n_aggr_state(p); ++s) {
            et_vec_critic[p][s] *= lambda_critic;
            if (s == curr_aggr_state[p])
                et_vec_critic[p][s] += 1;
            curr_v_pars[p][s] += lr * curr_td_error[p] * et_vec_critic[p][s] * (1-lambda_critic);
        }      
    }  
}


void MA_AC_ET::actor_update() {
    double aux_lr = lr_act(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++) {
        double lr = aux_lr * lr_act_factors[p];
        for (int s=0; s<(*env).n_aggr_state(p); ++s) {
            for (int a=0; a<curr_p_pars[p][s].size(); a++) {
                et_vec_actor[p][s][a] *= lambda_actor;
                if (s == curr_aggr_state[p]){
                    if (a == curr_action[p]) 
                        et_vec_actor[p][s][a] += 1 - curr_policy[p][s][a];
                    else
                        et_vec_actor[p][s][a] -= curr_policy[p][s][a];
                }
                curr_p_pars[p][s][a] += lr * curr_td_error[p] * et_vec_actor[p][s][a] * (1-lambda_actor) ;
            }  
        }
    }

    // If terminal state, the eligibity vectors are re-init to zero
    if (curr_info.done) {
        for (int p=0; p<(*env).n_players(); p++) {
            for (int s=0; s<(*env).n_aggr_state(p); ++s) {
                et_vec_critic[p][s] = 0;
                for (int a=0; a<(*env).n_actions(p,s); ++a)
                    et_vec_actor[p][s][a] = 0;
            }
        }
    }
}


void MA_NAC_AP_ET::child_init(){
    ap_par = vec3d(0);
    for (int p=0; p<(*env).n_players(); p++) {
        vec2d ap_par_of_player = vec2d(0);
        for (int s=0; s<(*env).n_aggr_state(p); ++s) 
            ap_par_of_player.push_back(vecd((*env).n_actions(p, s)));
        ap_par.push_back(ap_par_of_player);
    }
}


void MA_NAC_AP_ET::actor_update(){

    double aux_act_lr = lr_act(curr_step) * curr_gamma_fact;
    double aux_crit_lr = lr_crit(curr_step) * curr_gamma_fact;
    for (int p=0; p<(*env).n_players(); p++) {
        double lr_crit = aux_crit_lr * lr_crit_factors[p];
        double lr_act = aux_act_lr * lr_act_factors[p];
        for (int s=0; s<ap_par[p].size(); s++) {
            for (int a=0; a<curr_p_pars[p][curr_aggr_state[p]].size(); a++) {
                et_vec_actor[p][s][a] *= lambda_actor;
                if (s == curr_aggr_state[p]){
                    if (a == curr_action[p])
                        et_vec_actor[p][s][a] = 1 - curr_policy[p][s][a];
                    else
                        et_vec_actor[p][s][a] = - curr_policy[p][s][a];
                }
            }
            
            double aux_t = curr_td_error[p];
            for (int a=0; a<curr_p_pars[p][s].size(); a++)
                aux_t -= et_vec_actor[p][s][a] * ap_par[p][s][a];

            for (int a=0; a<curr_p_pars[p][s].size(); a++){
                ap_par[p][s][a] += lr_crit * et_vec_actor[p][s][a] * aux_t;
                curr_p_pars[p][s][a] += lr_act * ap_par[p][s][a] * (1-lambda_actor) ;
            }
            
            // Simplex Clipping
            int n_act = curr_p_pars[p][s].size();
            if (simpl_clip && (n_act > 1) ){
                vecd temp_policy = vecd(n_act);
                par2pol_boltzmann(curr_p_pars[p][s], temp_policy);
                // WORKING ONLY IF N_ACTIONS = 2
                for (int a=0; a < n_act; a++){
                    curr_policy[p][s][a] = std::max( std::min(temp_policy[a], (1-simpl_clip_eps*(n_act-1))), simpl_clip_eps);
                }
                pol2par_boltzmann(curr_policy[p][s], curr_p_pars[p][s]);
            }
        }
    }

}