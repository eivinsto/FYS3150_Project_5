#define _USE_MATH_DEFINES
#include "diffusion_equation_solver_1D.hpp"
#include "diffusion_equation_solver_2D.hpp"
#include <cmath>
#include <string>

// Initial state and boundary condition functions
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


int main(int argc, char** argv) {
  std::string dim = argv[1];                  // Specifies whether to run 1D or 2D simulation
  if (dim=="1D"){
    // 1D simulation
    int N = atoi(argv[2]);                    // Amount of steps in spatial coordinate
    double dt = atof(argv[3]);                // Timestep
    int M = atoi(argv[4]);                    // Amount of timesteps
    int write_limit = atoi(argv[5]);          // How often to write results to file
    std::string method = argv[6];             // Which scheme to use (1D)
    std::string output_filename = argv[7];    // Name of file to output results to
    double u_b = atof(argv[8]);               // Upper boundary condition (constant)
    double l_b = atof(argv[9]);               // Lower boundary condition (constant)

    // Generate solver object
    DiffusionEquationSolver1D system(N,dt,M,write_limit,init_func,method,output_filename,u_b,l_b);
    // Solve system
    system.solve();
  }

  if (dim=="2D"){
    // 2D simulation
    int N = atoi(argv[2]);                    // Amount of steps in spatial coordinates
    double dt = atof(argv[3]);                // Timestep
    int M = atoi(argv[4]);                    // Amount of timesteps
    int write_limit = atoi(argv[5]);          // How often to write results to file
    std::string output_filename = argv[6];    // Name of file to output results to
    std::string sim = argv[7];                // Specifies whether to perform general 2D simulation or lithosphere simulation ("heat")

    if (sim=="heat"){
      // Lithosphere temperature gradient simulation
      double ax = atof(argv[8]);                // Parameter used for rectangular system to be simulated with a square grid
      double ay = atof(argv[9]);                // Parameter used for rectangular system to be simulated with a square grid
      int M2 = atoi(argv[10]);                  // Amount of timesteps for second simulation
      int write_limit2 = atoi(argv[11]);        // How to often to write results to file in second simulation
      std::string output_filename2 = argv[12];  // Name of file to output results of second simulation to

      // Generate a solver object for first simulation
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func_heat,y_ub_heat,y_lb_heat,
                                       x_ub_heat,x_lb_heat,output_filename, unfertilized_source, ax, ay);
      // Run first simulation
      system.solve();

      // Update heat source term and set up second simulation with same object (current state of system kept)
      system.new_source_term(fertilized_source,output_filename2,M2,write_limit2);

      // Run second simulation
      system.solve();
    } else {
      // General 2D simulation

      // Generate a solver object
      DiffusionEquationSolver2D system(N,dt,M,write_limit,init_func2D,y_ub2D,y_lb2D,x_ub2D,x_lb2D,output_filename);

      // Check for extra command line argument (indicating that relative RMS error should be calculated)
      if (argc>8){
        std::string error_filename = argv[8];  // Name of file to output errors to

        // Initiate error calculation
        system.compare_with_analytic(analytic_2D, error_filename);
      }
      // Run simulation
      system.solve();
    }
  }
  return 0;
}

// Initial state for 1D simulations
double init_func(double x){
  return std::sin(2*M_PI*x) + x;
}

// Initial state for general 2D simulation
double init_func2D(double x, double y){
  return sin(2*M_PI*x)*sin(2*M_PI*y) + y ;
}

// Upper boundary condition in x-direction for general 2D simulation
double x_ub2D(double y){
  return y;
}

// Lower boundary condition in x-direction for general 2D simulation
double x_lb2D(double y){
  return y;
}

// Upper boundary condition in y-direction for general 2D simulation
double y_ub2D(double x){
  return 1;
}

// Lower boundary condition in y-direction for general 2D simulation
double y_lb2D(double x){
  return 0;
}

// Analytic solution for general 2D simulation
double analytic_2D(double x, double y, double t){
  return sin(2*M_PI*x)*sin(2*M_PI*y)*exp(-8*M_PI*M_PI*t) + y;
}

// Upper boundary in x-direction for lithosphere simulation
double x_ub_heat(double y){
  //return y*(1300-8) + 8;
  return -439.46*y*y + 1634.21*y + 81.40;
}

// Lower boundary in x-direction for lithosphere simulation
double x_lb_heat(double y){
  //return y*(1300-8) + 8;
  return -439.46*y*y + 1634.21*y + 81.40;
}

// Upper boundary in y-direction for lithosphere simulation
double y_ub_heat(double x){
  return 1300;
}

// Lower boundary in y-direction for lithosphere simulation
double y_lb_heat(double x){
  return 8;
}

// Source term function for lithosphere simulation (before further enrichment)
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

// Source term function for lithosphere simulation (after further enrichment)
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

// Initial state function for lithosphere simulation
double init_func_heat(double x, double y){
  return y*(1300-8) + 8;
}
