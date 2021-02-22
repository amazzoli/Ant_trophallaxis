#include "alg.h"


MARLAlgorithm::MARLAlgorithm(Environment* env, const param& params, std::mt19937& generator, bool verbose) :
env{env}, generator{generator}, verbose{verbose} {

    try {

        m_gamma = params.d.at("gamma");

        if (params.s.find("stop_by_discount") != params.s.end()) {
            if (params.s.at("stop_by_discount") == "true" || params.s.at("stop_by_discount") == "True")
                    stop_by_discount = true;
        }
        else
            stop_by_discount = false;

    } catch (std::exception) {
        throw std::invalid_argument( "Invalid algorithm parameters" );
    }

    unif_dist = std::uniform_real_distribution<double>(0.0,1.0);
}


void MARLAlgorithm::run(const param& params) {

    int n_steps, traj_points;
    save_return = true;
    save_env_info = true;
    // Reading parameters
    try {
        n_steps = params.d.at("n_steps");
        traj_points = params.d.at("traj_points");
        traj_step = round(n_steps/float(traj_points));
        if (params.s.find("save_returns") != params.s.end()) 
            if (params.s.at("save_returns") == "false" || params.s.at("save_returns") == "False")
                save_return = false;
        if (params.s.find("save_env_info") != params.s.end()) 
            if (params.s.at("save_env_info") == "false" || params.s.at("save_env_info") == "False")
                save_env_info = false;
    } catch (std::exception) {
        throw std::invalid_argument( "Invalid algorithm parameters" );
    }

    // Init    
    curr_episode = 1;
    curr_ep_step = 1;  
    curr_gamma_fact = 1;
    curr_aggr_state = veci((*env).n_players());
    curr_action = veci((*env).n_players());
    curr_info = env_info { vecd((*env).n_players()), false };
    curr_new_aggr_state = veci((*env).n_players());
    vecd ret = vecd((*env).n_players(), 0);
    vecd av_ret = vecd((*env).n_players());
    vecd t_reward = vecd((*env).n_players());
    return_traj = vec2d(0);
    ep_len_traj = veci(0);
    int ep_for_av_ret = 0;  
    int last_curr_step = 0;
    Perc perc(1, n_steps-1);

    // Env initialization   
    (*env).reset_state(curr_aggr_state);
    env_info_traj = vec2d(traj_points+1, vecd(0));
    int t_time = 0;

    // Algorithm-specific initialization
    init(params);

    // Main loop
    for (curr_step=0; curr_step<n_steps; ++curr_step){        

        // Algorithm-specific action at the current step
        get_action(curr_action);

        // std::cout << "s: ";
        // for(int p=0; p<(*env).n_players(); p++) {
        //     std::cout << (*env).aggr_state_descr()[p][curr_aggr_state[p]] << " ";
        // }
        // std::cout << " a: ";
        // for(int p=0; p<(*env).n_players(); p++)
        //     std::cout << (*env).action_descr()[p][curr_aggr_state[p]][curr_action[p]] << " ";

        // Envitonmental step
        int lrn_steps_elapsed = 1;
        (*env).step(curr_action, curr_info, lrn_steps_elapsed);
        for(int p=0; p<(*env).n_players(); p++)
            ret[p] += curr_info.reward[p] * curr_gamma_fact;
        (*env).aggr_state(curr_new_aggr_state);

        if (!stop_by_discount) curr_gamma_fact *= std::pow(m_gamma, lrn_steps_elapsed);

        // Stop with discount factor if enabled
        if (stop_by_discount) {
            int i = 0;
            bool stop_by_gamma = false;
            for (i; i<lrn_steps_elapsed; i++)
                if (unif_dist(generator) > m_gamma) {
                    curr_info.done = true;
                    (*env).gamma_stop(lrn_steps_elapsed-i-1);
                    stop_by_gamma = true;
                    break;
                }
            if (stop_by_gamma) lrn_steps_elapsed = i+1;
        }
        curr_ep_step += lrn_steps_elapsed;

        //std::cout << " step: " << curr_step;
        // for(int p=0; p<(*env).n_players(); p++){
        //    std::cout << curr_info.reward[p] << " ";
        // }
        // std::cout << " ls: " << lrn_steps_elapsed;
        // std::cout << "\n";

        // Algorithm-specific update
        learning_update(lrn_steps_elapsed);
        
        // Building the trajectory
        if (traj_step > 0 && curr_step%traj_step == 0) {
            build_traj();
            env_info_traj[t_time] = (*env).env_data();
            t_time++;
        }

        // At terminal state
        if (curr_info.done){ 

            // std::cout << "s': ";
            // for(int p=0; p<(*env).n_players(); p++) {
            //     std::cout << (*env).aggr_state_descr()[p][curr_new_aggr_state[p]] << " ";
            // }
            // std::cout << "DONE\n";
            
            // Updating return with last reward
            // (*env).terminal_reward(m_gamma, t_reward); 
            for(int p=0; p<(*env).n_players(); p++) {
                // ret[p] += curr_info.reward[p] * curr_gamma_fact;
                av_ret[p] += ret[p];
            }
            if (save_return){
                return_traj.push_back(ret);
                ep_for_av_ret++;
                ep_len_traj.push_back(curr_ep_step);
            }

            // Re-initialize the environment
            for(int p=0; p<(*env).n_players(); p++) ret[p] = 0;
            curr_ep_step = 1;
            curr_gamma_fact = 1;
            curr_episode++;
            (*env).reset_state(curr_aggr_state);
        } 
        // At non-terminal state
        else { 
            for(int p=0; p<(*env).n_players(); p++)
                curr_aggr_state[p] = curr_new_aggr_state[p];
        }

        // // Std output
        //std::cout << " average return over " << curr_step << " steps: ";
        if (verbose && perc.step(curr_step)) {
            if (ep_for_av_ret != 0){
                std::cout << " average return over " << ep_for_av_ret << " ep: ";
                for(int p=0; p<(*env).n_players(); p++)
                    std::cout << av_ret[p] / (float)ep_for_av_ret << " ";
            }
            else { // Continuous task
                std::cout << " average return at " << curr_step << " steps: ";
                for(int p=0; p<(*env).n_players(); p++){
                    std::cout << ret[p]/(curr_step - last_curr_step) << " ";   
                    ret[p] = 0;
                }
                last_curr_step = curr_step;
            }

            ep_for_av_ret = 0;
            for(double& r : av_ret) r = 0;
        }
    }
    build_traj();
    env_info_traj[t_time] = (*env).env_data();
}



void MARLAlgorithm::print_output(str dir) const {

    // Printing the returns and the episode lengths
    if (save_return) {
        std::ofstream file_r;
        file_r.open(dir + "return_traj.txt");

        file_r << "Episode_length\t";
        for(int p=0; p<(*env).n_players(); p++)
            file_r << "Return_p" << p+1 << "\t";
        file_r << "\n";

        for (int t=0; t<return_traj.size(); t++){
            file_r << ep_len_traj[t] << "\t";
            for(int p=0; p<(*env).n_players(); p++)
                file_r << return_traj[t][p] << "\t";
            file_r << "\n";
        }
        file_r.close();
    }

    // Printing the environment info trajectory
    if (save_env_info && traj_step > 0){
        if (env_info_traj.size() > 0 && env_info_traj[0].size() > 0)
            write_vec2d(env_info_traj, dir + "env_info_traj.txt", (*env).env_data_headers());
    }

    // Printing the algorithm-specific trajectories
    print_traj(dir);
}


MARLEval::MARLEval(Environment* env, const param& params, std::mt19937& generator, bool verbose) : 
MARLAlgorithm(env, params, generator, verbose) {}


void MARLEval::init(const param& params) {
    
    // Policy parameters init
    policy = vec3d((*env).n_players());
    if (params.s.find("init_pol_dir") != params.s.end()){
        std::cout << "Init policy from data: " << params.s.at("init_pol_dir") << "\n";
        for (int p=0; p<(*env).n_players(); p++) {
            policy[p] = read_vec2d( params.s.at("init_pol_dir") + "best_policy_" + std::to_string(p) + ".txt", true ); 
        }
    }
    else
        throw std::invalid_argument( "Invalid policy path for evaluation" );

    state_traj = vec2d();
    aggr_state_traj = vec2i();
    new_aggr_state_traj = vec2i();
    act_traj = vec2i();
    rew_traj = vec2d();
    done_traj = veci();
    time_traj = veci();
}


void MARLEval::get_action(veci& action) {
    // Lazy way.. we should build a vector of distributions...
    for (int p=0; p<(*env).n_players(); p++) {
        std::discrete_distribution<int> dist (
            policy[p][curr_aggr_state[p]].begin(), 
            policy[p][curr_aggr_state[p]].end()
        );
        action[p] = dist(generator);
    }
}


void MARLEval::build_traj() {
    state_traj.push_back((*env).state());
    aggr_state_traj.push_back(curr_aggr_state);
    new_aggr_state_traj.push_back(curr_new_aggr_state);
    act_traj.push_back(curr_action);
    rew_traj.push_back(curr_info.reward);
    done_traj.push_back(curr_info.done);
    time_traj.push_back(curr_ep_step);
}


void MARLEval::print_traj(str out_dir) const {
    write_vec2d(state_traj, out_dir + "ev_states.txt", (*env).state_descr());

    vecs ev_traj_header = vecs();
    for (int p=0; p<(*env).n_players(); p++)
        ev_traj_header.push_back("aggr_state" + std::to_string(p));
    for (int p=0; p<(*env).n_players(); p++)
        ev_traj_header.push_back("action" + std::to_string(p));
    for (int p=0; p<(*env).n_players(); p++)
        ev_traj_header.push_back("aggr_new_state" + std::to_string(p));
    for (int p=0; p<(*env).n_players(); p++)
        ev_traj_header.push_back("reward" + std::to_string(p));
    ev_traj_header.push_back("done");
    ev_traj_header.push_back("step");

    vec2d ev_traj = vec2d();
    for (int t=0; t<aggr_state_traj.size(); t++){
        vecd aux_traj = vecd();
        for (const int& as : aggr_state_traj[t])
            aux_traj.push_back(as);
        for (const int& a : act_traj[t])
            aux_traj.push_back(a);
        for (const int& ans : new_aggr_state_traj[t])
            aux_traj.push_back(ans);
        for (const double& r : rew_traj[t])
            aux_traj.push_back(r);
        aux_traj.push_back(done_traj[t]);
        aux_traj.push_back(time_traj[t]);
        ev_traj.push_back(aux_traj);
    }

    write_vec2d(ev_traj, out_dir + "ev_info.txt", ev_traj_header);
}