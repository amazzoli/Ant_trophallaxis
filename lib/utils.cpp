#include "utils.h"
#include <cctype>


void par2pol_boltzmann(const vecd& params, vecd& policy){
    double max = *std::max_element(params.begin(), params.end());
    double norm = 0;
    for (int i=0; i<params.size(); i++){
        double val = exp(params[i]-max);
        policy[i] = val;
        norm += val;
    }
    for (int i=0; i<policy.size(); i++) policy[i] /= norm;
}

/*
void pol2par_boltzmann(const vecd& policy, vecd& params){
    for (int i=0; i<params.size(); i++){
        double val = 0;
        if (policy[i] > 10E-20)
            params[i] = log(policy[i]);
        else
            params[i] = -20;
    }
}
*/


void pol2par_boltzmann(const vecd& policy, vecd& params){
    double norm = 0;
    for (int i=0; i<params.size(); i++){
        if (policy[i] > 10E-20)
            norm += policy[i]*log(policy[i]);
    }
    
    for (int i=0; i<params.size(); i++){
        double val = 0;
        if (policy[i] > 10E-20)
            params[i] = log(policy[i]) - norm ;
        else
            params[i] = -20;
    }
}


double plaw_dacay(double t, double t_burn, double expn, double a0, double ac){
    if (t < t_burn) 
        return a0;
    else {
        return a0 * ac / (ac + pow(t-t_burn, expn));
    }
}


vecd str2vecd(str line, str separator, bool sep_at_end) {
    std::size_t sep_pos = line.find(separator);
    if (sep_pos == std::string::npos) {
        if (sep_at_end)
            throw std::runtime_error(separator + " separator not found in " + line);
        else {
            vecd v = {std::stod(line.substr(0, sep_pos))};
            return v;
        }
    }

    str elem = line.substr(0, sep_pos);
    vecd v = vecd(0);
    try {
        v.push_back(std::stod(elem));
    }
    catch (std::exception& e){
        v.push_back(0);
    }

    while (true){
        std::size_t next_sep_pos = line.find(separator, sep_pos+1);
        if (sep_at_end && next_sep_pos == std::string::npos) break;
        str elem = line.substr(sep_pos+1, next_sep_pos-sep_pos);
        try{
            v.push_back(std::stod(elem));
        }
        catch (std::exception& e){
            v.push_back(0);
        }
        if (!sep_at_end && next_sep_pos == std::string::npos) break;
        sep_pos = next_sep_pos;
    }

    return v;
}


vec2d str2vec2d(str line, str separator1, str separator2, bool sep_at_end) {
    throw std::runtime_error("str2vec2d to implement");
    return vec2d(0);
}


param parse_param_file(str file_path){

    dictd paramd;
    dictvecd paramvecd;
    dicts params;
    
    std::ifstream param_file (file_path);
    if (!param_file.is_open())
        throw std::runtime_error("Error in opening the parameter file at "+file_path);

    str line;
    while ( getline (param_file, line) ) {
        std::size_t tab_pos = line.find("\t");
        str key = line.substr(0,tab_pos);
        str value = line.substr(tab_pos+1, str::npos);

        std::size_t comma_pos = value.find(",");
        if (value.find(",") != str::npos){
            paramvecd[key] = str2vecd(value, ",", true); // Parse a vector
        }
        else{
            try {
                double vald = std::stod(value); // Parse a double
                paramd[key] = vald;
            } catch (std::invalid_argument){
                if (isspace(value[value.size()-1]))
                    value = value.substr(0,value.size()-1);
                params[key] = value; // Parse a string if stod gives exception
            }
        }
    }
    param_file.close();

    if (paramd.size() == 0 && paramvecd.size() == 0 && params.size() == 0)
        throw std::runtime_error("Empty parameter file");

    return param{paramd, paramvecd, params};
}


vecd read_vecd(str file_path) {

    vecd val(0);
    std::ifstream file (file_path);
    if (!file.is_open())
        throw std::runtime_error("Error in opening the file at "+file_path);

    str line;
    try {
        while ( getline (file, line) )
            val.push_back(std::stod(line));
    } catch (std::exception) {
        throw std::runtime_error("Error in reading the file at "+file_path);
    }

    file.close();

    return val;
}


void write_vecd(const vecd& v, const str file_path) {
    std::ofstream out;
    out.open(file_path);

    for (int i=0; i<v.size(); i++)
        out << v[i] << "\n";

    out.close();
}



vec2d read_vec2d(str file_path, bool sep_at_end) {

    vec2d v(0);
    std::ifstream file (file_path);
    if (!file.is_open())
        throw std::runtime_error("Error in opening the file at "+file_path);

    str line;
    try {
        while ( getline (file, line) )
            v.push_back(str2vecd(line, "\t", sep_at_end));
    } catch (std::exception) {
        throw std::runtime_error("Error in reading the file at "+file_path);
    }

    file.close();

    return v;
}


void write_vec2d(const vec2d& v, const str file_path, const vecs& header) {
    std::ofstream out;
    out.open(file_path);

    if (header.size() != 0){
        for (int i=0; i<header.size(); i++)
            out << header[i] << "\t";
        out << "\n";
    }

    for (int i=0; i<v.size(); i++){
        for (int j=0; j<v[i].size(); j++)
            out << v[i][j] << "\t";
        out << "\n";
    }
    
    out.close();
}


vec3d read_vec3d(str file_path) {

    vec3d v(0);
    std::ifstream file (file_path);
    if (!file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    str line;
    try {
        while ( getline (file, line) )
            v.push_back(str2vec2d(line, ",", "\t", false));
    } catch (std::exception) {
        throw std::runtime_error("Error in reading the file at "+file_path);
    }

    file.close();

    return v;
}


void write_vec3d(const vec3d& v, const str file_path, const vec2s& header) {
    std::ofstream out;
    out.open(file_path);

    if (header.size() != 0){
        for (int i=0; i<header.size(); i++){
            for (int j=0; j<header[i].size(); j++)
                out << header[i][j] << ",";
            out << "\t";
        }
        out << "\n";
    }

    for (int i=0; i<v.size(); i++){
        for (int j=0; j<v[i].size(); j++){
            for (int k=0; k<v[i][j].size(); k++)
                out << v[i][j][k] << ",";
            out << "\t";
        }
        out << "\n";
    }
    
    out.close();
}