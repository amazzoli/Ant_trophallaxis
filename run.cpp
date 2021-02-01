#include "lib/nac.h"
//#include "../lib/qalg.h"
//#include "../lib/eval.h"
#include "lib/ants.h"


// Algorithm launcher. It must be launched giving also the environment_name and the trial_name.
// The first refers to the environment that is trained, the second to the algorithm that is launched.


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator);
MARLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator);


int main(int argc, char** argv) {

    if (argc != 3)
        throw std::runtime_error("Two strings must be passed during execution: environment name and run name");

    // Init random generator
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 1;
    std::cout << "\nWarning! Fixed seed!\n\n";
    std::mt19937 generator(seed);

    // Importing the parameters file
    std::string env_name(argv[1]), alg_name(argv[2]), data_dir = "data/";
    param params_env = parse_param_file(data_dir+env_name+"/"+alg_name+"/param_env.txt"); // Def in utils

    // Constructing the environment.
    Environment* env = get_env(env_name, params_env, generator);
    std::cout << "Environment successfully built:\n" << (*env).descr() << "\n\n";

    // // Constructing the algorithm
    param alg_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_alg.txt");
    MARLAlgorithm* alg = get_alg(env, alg_params, generator);
    std::cout << "Algorithm successfully initialized:\n" << (*alg).descr() << "\n\n";

    // Running the algorithm
    Timer timer;
    std::cout << "Algorithm started\n";
    (*alg).run(alg_params);
    std::cout << "\nAlgorithm completed in " << timer.elapsed() << " seconds\n";

    // Printing the trajectories
    (*alg).print_output(data_dir + env_name + "/" + alg_name + "/");
    std::cout << "Trajectories successfully printed at " << data_dir + env_name + "/" + alg_name + "/" << "\n";

    delete alg;

    // Evaluation
    if (alg_params.d.find("eval_steps") != alg_params.d.end() && alg_params.d.at("eval_steps") > 0) {
        std::cout << "\nEvaluation started\n";
        MARLEval eval = MARLEval(env, alg_params, generator, false);
        alg_params.d.at("n_steps") = alg_params.d.at("eval_steps");
        alg_params.d.at("traj_points") = alg_params.d.at("n_steps");
        alg_params.s["init_pol_dir"] = data_dir+env_name+"/"+alg_name + "/";
        alg_params.s["save_return"] = "false";
        alg_params.s["save_env_info"] = "false";
        eval.run(alg_params);
        std::cout << "Evaluation completed\n";
        eval.print_output(data_dir + env_name + "/" + alg_name + "/");
        std::cout << "Trajectories successfully printed\n";
    }

    delete env;

    return 0;
}


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator) {
    if (env_name == "ant_ma"){
        return new Ants_ma(params, generator);
    }
    if (env_name == "ant_cons"){
        return new Ants_consume(params, generator);
    }
    if (env_name == "ant_cons2" || env_name == "ant_cons2_fast"){
        return new Ants_consume2(params, generator);
    }
    else throw std::invalid_argument( "Invalid environment name" );
}


MARLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator) {

    std::string alg_name = params.s.at("alg_type");

    if (alg_name == "ac"){
        return new MA_AC(env, params, generator);
    }
    else if (alg_name == "ac_et"){
        return new MA_AC_ET(env, params, generator);
    }
    else if (alg_name == "nac"){
		return new MA_NAC_AP(env, params, generator);
    }
    else if (alg_name == "nac_et"){
        return new MA_NAC_AP_ET(env, params, generator);
    }
    else throw std::invalid_argument( "Invalid algorithm name" );
}
