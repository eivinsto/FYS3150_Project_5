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
double init_func_heat(double, double);
double unfertilized_source(double, double, double);
double fertilized_source(double, double, double);
double x_ub_heat(double);
double x_lb_heat(double);
double y_ub_heat(double);
double y_lb_heat(double);


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

    std::string sim = argv[7];
    if (sim=="heat"){
      double ax = atof(argv[8]);
      double ay = atof(argv[9]);
      std::string sim2 = argv[10];
      if (sim2=="enriched"){
        DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                         x_ub_heat,x_lb_heat,output_filename, fertilized_source, ax, ay);
      } else {
        DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                         x_ub_heat,x_lb_heat,output_filename, unfertilized_source, ax, ay);
      }
    } else {
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func2D,y_ub2D,y_lb2D,x_ub2D,x_lb2D,output_filename);
      system.solve();
    }
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

double x_ub_heat(double y){
  return y*(1300-8) + 8;
}

double x_lb_heat(double y){
  return y*(1300-8) + 8;
}

double y_ub_heat(double x){
  return 1300;
}

double y_lb_heat(double x){
  return 8;
}

double unfertilized_source(double x, double y, double t){
  // Returned heat is in units K Gy^-1
  if (y <= 80/120){
    return 0.44923;
  } else if ((y > 80/120) && (y <= 100/120)) {
    return 3.1446;
  } else {
    return 12.578;
  }
}

double fertilized_source(double x, double y, double t){
  // Returned heat is in units K Gy^-1
  if (y <= 80/120){
    return 0.44923 + 4.4923*(0.4*std::exp(-0.155*t) + 0.4*std::exp(-0.0495*t) + 0.2*std::exp(-0.555*t));
  } else if ((y > 80/120) && (y <= 100/120)) {
    return 3.1446;
  } else {
    return 12.578;
  }
}

double init_func_heat(double x, double y){
  return y*(1300-8) + 8;
}
