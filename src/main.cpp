#include "diffusion_equation_solver.hpp"
#include <cmath>
#include <string>

double init_func(double);

int main(int argc, char** argv) {
  int N = atoi(argv[1]);
  double dt = atof(argv[2]);
  int M = atoi(argv[3]);
  int write_limit = atoi(argv[4]);
  std::string method = argv[5];
  std::string output_filename = argv[6];

  DiffusionEquationSolver system(N,dt,init_func,method,output_filename);
  system.solve(M,write_limit);

  return 0;
}

double init_func(double x){
  return 100*std::exp(-10*x);
}
