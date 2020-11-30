#include "diffusion_equation_solver.hpp"
#include <cmath>
#include <string>

double init_func(double);

int main() {
  int N = 100;
  double dt = 1e-5;
  std::string method = "ForwardEuler";
  std::string output_filename = "../data/test.dat";
  int M = 1000;
  int write_limit = 100;

  DiffusionEquationSolver system(N,dt,init_func,method,output_filename);
  system.solve(M,write_limit);

  return 0;
}

double init_func(double x){
  return 100*std::exp(-10*x);
}
