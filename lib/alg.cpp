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
    // Reading parameters
    try {
        n_steps = params.d.at("n_steps");
        traj_points = params.d.at("traj_points");
        traj_step = round(n_steps/float(traj_points));
        if (params.s.find("save_returns") != params.s.end()) 
            if (params.s.at("save_returns") == "false" || params.s.at("save_returns") == "False")
                save_return = false;
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
    int ep_for_av_ret = 1;  
    Perc perc(10, n_steps-1);

    // Env initialization   
    (*env).reset_state(curr_aggr_state);
    env_info_traj = vec2d(traj_points+1, vecd(0));
    int t_time = 0;

    // Algorithm-specific initialization
    init(params);

    // Main loop
    for (curr_step=0; curr_step<n_steps; ++curr_step){

        // std::cout << "s: ";
        // for(int p=0; p<(*env).n_players(); p++) {
        //     std::cout << (*env).aggr_state_descr()[p][curr_aggr_state[p]] << " ";
        // }
        
        // std::cout << ". a: ";
        // Algorithm-specific action at the current step
        get_action(curr_action);
        // for(int p=0; p<(*env).n_players(); p++)
        //     std::cout << (*env).action_descr()[p][curr_aggr_state[p]][curr_action[p]] << " ";

        // Envitonmental step
        int lrn_steps_elapsed = 1;
        (*env).step(curr_action, curr_info, lrn_steps_elapsed);
        for(int p=0; p<(*env).n_players(); p++)
            ret[p] += curr_info.reward[p] * curr_gamma_fact;
        (*env).aggr_state(curr_new_aggr_state);

        if (!stop_by_discount) curr_gamma_fact *= std::pow(m_gamma, lrn_steps_elapsed);

        // std::cout << ". r: ";
        // for(int p=0; p<(*env).n_players(); p++){
        //    std::cout << curr_info.reward[p] << " ";
        // }
        // std::cout << "\n";

        // Stop with discount factor if enabled
        if (stop_by_discount) {
            for (int = 0; i<lrn_steps_elapsed; i++)
                if (unif_dist(generator) > m_gamma)
                    curr_info.done = true;
        }

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
            // std::cout << "DONE\n";
            // Updating return with last reward
            (*env).terminal_reward(m_gamma, t_reward); 
            for(int p=0; p<(*env).n_players(); p++) {
                ret[p] += t_reward[p] * curr_gamma_fact;
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
            curr_ep_step++;
        }

        // Std output
        if (verbose && perc.step(curr_step)) {
            if (ep_for_av_ret != 0){
                std::cout << " average return over " << ep_for_av_ret << " ep: ";
                for(int p=0; p<(*env).n_players(); p++)
                    std::cout << av_ret[p] / (float)ep_for_av_ret << " ";
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
    if (traj_step > 0){
        if (env_info_traj.size() > 0 && env_info_traj[0].size() > 0)
            write_vec2d(env_info_traj, dir + "env_info_traj.txt", (*env).env_data_headers());
    }

    // Printing the algorithm-specific trajectories
    print_traj(dir);
}


