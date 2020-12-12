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
double analytic_2D(double, double, double);
double init_func_heat(double, double);
double unfertilized_source(double, double, double);
double fertilized_source(double, double, double);
double x_ub_heat(double);
double x_lb_heat(double);
double y_ub_heat(double);
double y_lb_heat(double);
double analytic_heat(double, double, double);


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

    /*
    if (sim=="heat"){
      double ax = atof(argv[8]);
      double ay = atof(argv[9]);
      std::string source_type = argv[10];
      if (source_type=="enriched"){
        DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                         x_ub_heat,x_lb_heat,output_filename, fertilized_source, ax, ay);
        system.solve();
      } else {
        DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                         x_ub_heat,x_lb_heat,output_filename, unfertilized_source, ax, ay);
        if (argc>11){
          std::string error_filename = argv[11];
          system.compare_with_analytic(analytic_heat, error_filename);
        }
        system.solve();
      }
    } else {
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func2D,y_ub2D,y_lb2D,x_ub2D,x_lb2D,output_filename);
      if (argc>8){
        std::string error_filename = argv[8];
        system.compare_with_analytic(analytic_2D, error_filename);
      }
      system.solve();
    }
    */
    if (sim=="heat"){
      double ax = atof(argv[8]);
      double ay = atof(argv[9]);
      int M2 = atoi(argv[10]);
      int write_limit2 = atoi(argv[11]);
      std::string output_filename2 = argv[12];
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                       x_ub_heat,x_lb_heat,output_filename, unfertilized_source, ax, ay);
      system.solve();
      system.new_source_term(fertilized_source,output_filename2,M2,write_limit2);
      system.solve();
    } else {
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func2D,y_ub2D,y_lb2D,x_ub2D,x_lb2D,output_filename);
      if (argc>8){
        std::string error_filename = argv[8];
        system.compare_with_analytic(analytic_2D, error_filename);
      }
      system.solve();
    }
  }
  return 0;
}

double init_func(double x){
  return std::sin(2*M_PI*x) + x;
}

double init_func2D(double x, double y){
  return sin(2*M_PI*x)*sin(2*M_PI*y) + y ;
}

double x_ub2D(double y){
  return y;
}

double x_lb2D(double y){
  return y;
}

double y_ub2D(double x){
  return 1;
}

double y_lb2D(double x){
  return 0;
}

double analytic_2D(double x, double y, double t){
  return sin(2*M_PI*x)*sin(2*M_PI*y)*exp(-8*M_PI*M_PI*t) + y;
}

double x_ub_heat(double y){
  return y*(1300-8) + 8;
}

double x_lb_heat(double y){
  return y*(1300-8) + 8;
};

double y_ub_heat(double x){
  return 1300;
}

double y_lb_heat(double x){
  return 8;
}

double unfertilized_source(double x, double y, double t){
  // Returned heat is in units K Gy^-1
  if (y >= (40.0/120.0)){
    return 450.0;
  } else if ((y < (40.0/120.0)) && (y >= (20.0/120.0))) {
    return 3150.0;
  } else {
    return 12600.0;
  }
}

double fertilized_source(double x, double y, double t){
  // Returned heat is in units K Gy^-1
  if (y >= (40.0/120.0)){
    double Q = 450.0;
    if ( (x>=(75.0/300.0)) && (x<=(225.0/300.0)) ){
      Q += 4500.0*(0.4*std::exp(-0.155*t) + 0.4*std::exp(-0.0495*t) + 0.2*std::exp(-0.555*t));
    }
    return Q;
  } else if ((y < (40.0/120.0)) && (y >= (20.0/120.0))) {
    return 3150.0;
  } else {
    return 12600.0;
  }
}

double init_func_heat(double x, double y){
  return y*(1300-8) + 8;
}

double analytic_heat(double x, double y, double t){
  return init_func_heat(x,y);
}
