#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "math.h"


using str = std::string;
using vecs = std::vector<str>;
using vec2s = std::vector<vecs>;
using vec3s = std::vector<vec2s>;
using vecf = std::vector<float>;
using veci = std::vector<int>;
using vec2i = std::vector<veci>;
using vecd = std::vector<double>;
using vec2d = std::vector<vecd>;
using vec3d = std::vector<vec2d>;
using vec4d = std::vector<vec3d>;
using dictd = std::map<str, double>;
using dictvecd = std::map<str, vecd>;
using dicts = std::map<str, str>;
using d_i_fnc = std::function<double(int)>;


/* Parameters. They can be doubles, vector of doubles or strings */
struct param {
	dictd d;
	dictvecd vecd;
	dicts s;
};



void par2pol_boltzmann(const vecd& params, vecd& policy);

void pol2par_boltzmann(const vecd& policy, vecd& params);

double plaw_dacay(double t, double t_burn, double expn, double a0, double ac);

vecd str2vecd(str line, str separator, bool sep_at_end);

vec2d str2vec2d(str line, str separator1, str separator2, bool sep_at_end);

vecd parse_str_of_doubles(str a_str);

param parse_param_file(str file_path);

vecd read_vecd(str file_path);

vec2d read_vec2d(str file_path, bool sep_at_end=false);

vec3d read_vec3d(str file_path);

void write_vecd(const vecd& v, const str file_path);

void write_vec2d(const vec2d& v, const str file_path, const vecs& header=vecs(0));

void write_vec3d(const vec3d& v, const str file_path, const vec2s& header=vec2s(0));


/* Class for measuring the time between the reset and en enlapsed call */
class Timer {
	private:
		using clock_t = std::chrono::high_resolution_clock;
		using second_t = std::chrono::duration<double, std::ratio<1> >;
		std::chrono::time_point<clock_t> m_beg;
	public:
		Timer() : m_beg(clock_t::now()) { };
		/* Set the onset for the time measure */
		void reset() { m_beg = clock_t::now(); }
		/* Get the time in seconds enlapsed from reset */
		double elapsed() const { return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count(); }
};


class Perc {
	private:
		int m_perc_step;
		int m_max_steps;
		double m_last_perc;
	public:
		Perc(int perc_step, int max_steps) : m_max_steps{max_steps}, m_last_perc{0.0} {
			if (perc_step<1 || perc_step>100) {
				std::cout << "Invalid percentage step. Set by default to 10%\n";
				m_perc_step = 10;
			}
			else m_perc_step = perc_step;
		};
		bool step(int curr_step) {
			double perc = (double)curr_step/(double)m_max_steps*100;
			if (perc >= m_last_perc){
				std::cout << "\n" << round(perc) << "%";
				m_last_perc = round(perc) + m_perc_step;
				return true;
			}
			return false;
		}
};

#endif