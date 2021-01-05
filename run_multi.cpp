#include "lib/nac.h"
#include "lib/ants.h"


// Algorithm launcher. It must be launched giving also the environment_name and the trial_name.
// The first refers to the environment that is trained, the second to the algorithm that is launched.


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator);
MARLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator, bool verbose);


int main(int argc, char** argv) {

    if (argc != 3)
        throw std::runtime_error("Two strings must be passed during execution: environment name and run name");

    // Init random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Importing the parameters file
    std::string env_name(argv[1]), alg_name(argv[2]), data_dir = "data/";
    param param_count = parse_param_file(data_dir+env_name+"_multi/"+alg_name+"/info.txt");

    Environment* env;
    MARLAlgorithm* alg;
    param params_env, alg_params;
    for (int i=0; i<param_count.d.at("counts"); i++) {
        params_env = parse_param_file(data_dir+env_name+"_multi/"+alg_name+"/"+std::to_string(i)+"_param_env.txt");
        alg_params = parse_param_file(data_dir + env_name + "_multi/" + alg_name + "/"+std::to_string(i)+"_param_alg.txt");
        
        // Constructing the environment.
        Environment* env = get_env(env_name, params_env, generator);

        // Constructing the algorithm
        MARLAlgorithm* alg = get_alg(env, alg_params, generator, false);

        // Running the algorithm
        Timer timer;
        (*alg).run(alg_params);

        // Printing the trajectories
        (*alg).print_output(data_dir + env_name + "_multi/" + alg_name + "/"+std::to_string(i)+"_");
        std::cout << i << "/" << param_count.d.at("counts") << "\n";

        delete env;
        delete alg;
    }

    return 0;
}


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator) {
    if (env_name == "ant_ma"){
        return new Ants_ma(params, generator);
    }
    if (env_name == "ant_cons"){
        return new Ants_consume(params, generator);
    }
    else throw std::invalid_argument( "Invalid environment name" );
}


MARLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator, bool verbose) {

    std::string alg_name = params.s.at("alg_type");

    if (alg_name == "ac"){
        return new MA_AC(env, params, generator, verbose);
    }
    else if (alg_name == "nac"){
		return new MA_NAC_AP(env, params, generator, verbose);
    }
    else throw std::invalid_argument( "Invalid algorithm name" );
}
