#define _USE_MATH_DEFINES
#include "diffusion_equation_solver_1D.hpp"
#include "diffusion_equation_solver_2D.hpp"
#include <cmath>
#include <string>

double init_func(double);
double init_func2D(double, double);
double x_ub2D(double);
double x_lb2D(double);
double y_ub2D(double);
double y_lb2D(double);


int main(int argc, char** argv) {
  std::string dim = argv[1];
  if (dim=="1D"){
    int N = atoi(argv[2]);
    double dt = atof(argv[3]);
    int M = atoi(argv[4]);
    int write_limit = atoi(argv[5]);
    std::string method = argv[6];
    std::string output_filename = argv[7];
    double u_b = atof(argv[8]);
    double l_b = atof(argv[9]);

    DiffusionEquationSolver1D system(N,dt,M,write_limit,init_func,method,output_filename,u_b,l_b);
    system.solve();
  }

  if (dim=="2D"){
    int N = atoi(argv[2]);
    double dt = atof(argv[3]);
    int M = atoi(argv[4]);
    int write_limit = atoi(argv[5]);
    std::string output_filename = argv[6];

    DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func2D,y_ub2D,y_lb2D,x_ub2D,x_lb2D,output_filename);
    system.solve();
  }

  return 0;
}

double init_func(double x){
  return std::sin(2*M_PI*x) + x;
}

double init_func2D(double x, double y){
  return x*y;
}

double x_ub2D(double y){
  return 1;
}

double x_lb2D(double y){
  return 0;
}

double y_ub2D(double x){
  return x;
}

double y_lb2D(double x){
  return x;
}
