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
  double u_b = atof(argv[7]);
  double l_b = atof(argv[8]);


  DiffusionEquationSolver system(N,dt,M,write_limit,init_func,method,output_filename,u_b,l_b);
  system.solve();

  return 0;
}

double init_func(double x){
  return std::exp(x);
}
