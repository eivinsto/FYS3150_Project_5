#include <string>
#include <iostream>
#include <iomanip>
#include "diffusion_equation_solver_2D.hpp"
#include <fstream>
#include <armadillo>

DiffusionEquationSolver2D::DiffusionEquationSolver2D(int N, double dt, int M, int write_limit,
                                                     double (*init_func)(double, double),
                                                     double (*y_ub)(double), double (*y_lb)(double),
                                                     double (*x_ub)(double), double (*x_lb)(double),
                                                     std::string ofilename){

  m_init_func = init_func;                               // Initial condition function
  m_y_ub = y_ub;                                         // Upper boundary function in y-direction
  m_y_lb = y_lb;                                         // Lower boundary function in y-direction
  m_x_ub = x_ub;                                         // Upper boundary function in x-direction
  m_x_lb = x_lb;                                         // Lower boundary function in x-direction
  m_N = N;                                               // Amount of lengthsteps
  m_dt = dt;                                             // Timestep
  m_M = M;                                               // Amount of timesteps
  m_write_limit = write_limit;                           // Write results every <write_limit> steps
  m_ofilename = ofilename;                               // Output filename
  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of output file
}

void DiffusionEquationSolver2D::jacobi(){
  return;
}

void DiffusionEquationSolver2D::solve(){
  return;
}
