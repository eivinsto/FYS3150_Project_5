#include <string>
#include <iostream>
#include <iomanip>
#include "diffusion_equation_solver_2D.hpp"
#include <fstream>
#include <armadillo>
#include <cmath>

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
  m_ofilename = ofilename;                               // Output filename
  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of output file
  m_h = 1.0/double(m_N+1);                                       // Lengthstep
  m_alpha = m_dt/(m_h*m_h);                              // Coefficient for use in time integration
  m_u = arma::zeros<arma::mat>(m_N,m_N);                            // Solution matrix
  m_q = arma::zeros<arma::mat>(m_N,m_N);                            // Source term
  m_write_limit = write_limit;                           // Write every <write_limit> timesteps
  m_alpha_coeff = 1.0/(1.0+4.0*m_alpha);
}

void DiffusionEquationSolver2D::jacobi(){
  // Generate dense matrix to store previous solution
  arma::mat old = arma::ones<arma::mat>(m_N,m_N);
  double s = 0;

  // Boundary conditions
  for (int i = 1; i < m_N-1; i++){
    m_u(0,i) = m_x_lb(i*m_h);
    m_u(m_N-1,i) = m_x_ub(i*m_h);
    m_u(i,0) = m_y_lb(i*m_h);
    m_u(i,m_N-1) = m_y_ub(i*m_h);
  }
  m_u(0,0) = m_x_lb(0);
  m_u(m_N-1,m_N-1) = m_x_ub(1);
  m_u(m_N-1,0) = m_y_ub(1);
  m_u(0,m_N-1) = m_y_lb(0);

  // Iterative solver
  for (int k = 0; k < m_maxiter; k++){
    for (int i = 1; i < m_N-1; i++){
      for (int j = 1; j < m_N-1; j++){
        m_u(i,j) = m_alpha_coeff*(m_alpha*(old(i+1,j) + old(i-1,j) + old(i,j-1) + old(i,j+1)) + m_dt*m_q(i,j));
      }
    }

    // Check convergence
    s = 0;
    double term = 0;
    for (int i = 0; i < m_N; i++){
      for (int j = 0; j < m_N; j++){
        term = old(i,j) - m_u(i,j);
        s += term*term;

        // Overwrite old solution
        old(i,j) = m_u(i,j);
      }
    }
    if (std::sqrt(s) < m_abstol){
      // Return if solution has converged
      return;
    }
  }
  // Output error/warning if solution did not converge within set number of max iterations
  std::cerr << "Solution using Jacobi iterative method did not converge properly within set limit of maximum iterations." << std::endl;
  std::cout << "Final sum: " << s << std::endl;
}

void DiffusionEquationSolver2D::solve(){
  // Set initial condition
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_u(i,j) = m_init_func(i*m_h,j*m_h);
    }
  }
  // TEMPORARY TO MAKE SURE IT WORKS AS IT SHOULD (should be ONLY in jacobi() function)
  // Boundary conditions
  for (int i = 1; i < m_N-1; i++){
    m_u(0,i) = m_x_lb(i*m_h);
    m_u(m_N-1,i) = m_x_ub(i*m_h);
    m_u(i,0) = m_y_lb(i*m_h);
    m_u(i,m_N-1) = m_y_ub(i*m_h);
  }
  m_u(0,0) = m_x_lb(0);
  m_u(m_N-1,m_N-1) = m_x_ub(1);
  m_u(m_N-1,0) = m_y_ub(1);
  m_u(0,m_N-1) = m_y_lb(0);


  // Time iteration
  for (m_t = 0; m_t < m_M; m_t++){
    m_q = m_u;


    // TEMPORARY PLACEMENT TO CHECK BOUNDARY CONDITIONS (should be after call to jacobi())
    if (m_t%m_write_limit == 0){
      write_to_file();
    }

    // Move one step in time
    jacobi();

  }
}

void DiffusionEquationSolver2D::write_to_file(){
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_ofile << std::setw(15) << std::setprecision(8) << m_u(i,j) << ' ';
    }
  }
  m_ofile << std::setw(15) << std::setprecision(8) << m_t*m_dt << std::endl;
}