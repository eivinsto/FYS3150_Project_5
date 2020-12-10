#include <string>
#include <iostream>
#include <iomanip>
#include "diffusion_equation_solver_2D.hpp"
#include <fstream>
#include <armadillo>
#include <cmath>
#include <omp.h>

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
  m_N = N+1;                                               // Amount of lengthsteps
  m_dt = dt;                                             // Timestep
  m_M = M;                                               // Amount of timesteps
  m_ofilename = ofilename;                               // Output filename
  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of output file
  m_h = 1.0/double(N);                                       // Lengthstep
  m_alpha = m_dt/(m_h*m_h);                              // Coefficient for use in time integration
  m_u = arma::zeros<arma::mat>(m_N,m_N);                            // Solution matrix
  m_q = arma::zeros<arma::mat>(m_N,m_N);                            // Source term
  m_write_limit = write_limit;                           // Write every <write_limit> timesteps
  m_diag_element = 1.0/(1.0+4.0*m_alpha);
}

DiffusionEquationSolver2D::DiffusionEquationSolver2D(int N, double dt, int M, int write_limit,
                                                     double (*init_func)(double, double),
                                                     double (*y_ub)(double), double (*y_lb)(double),
                                                     double (*x_ub)(double), double (*x_lb)(double),
                                                     std::string ofilename, double (*source_term)(double,double,double),
                                                     double ax, double ay)
                         : DiffusionEquationSolver2D(N, dt, M, write_limit, init_func, y_ub, y_lb,
                                                     x_ub, x_lb, ofilename){
  m_use_source_term = true;                             // Specify that source term function has been provided
  m_source_term = source_term;                          // Store source term function
  m_Ax = 1.0/(ax*ax);                                   // Calculate constant for x-direction used for rectangular grid
  m_Ay = 1.0/(ay*ay);                                   // Calculate constant for y-direction used for rectangular grid
  m_diag_element = 1.0/(1.0 + 2*m_alpha*(m_Ax + m_Ay)); // Specify diagonal elements in terms of Ax and Ay
}

void DiffusionEquationSolver2D::jacobi(){
  // Define sum variable
  double s = 1;

  // Boundary conditions
  for (int i = 0; i < m_N; i++){
    m_u(0,i) = m_x_lb(i*m_h);
    m_u(m_N-1,i) = m_x_ub(i*m_h);
    m_u(i,0) = m_y_lb(i*m_h);
    m_u(i,m_N-1) = m_y_ub(i*m_h);
  }

  // Generate dense matrix to store previous solution
  arma::mat old = m_u; // Initial guess is value in previous timestep

  int k = 0;
  int i, j;
  // int thrds = 0.5*omp_get_max_threads();
  // Iterative solver
  while (std::sqrt(s) > m_abstol && k < m_maxiter){
    // #pragma omp parallel if(m_N > 1000) num_threads(2) default(shared) private(i, j) firstprivate(m_diag_element, m_alpha, m_Ax, m_Ay) reduction(+:s)
    {
      // Define elements used for extraction
      double u10;
      double u20;
      double u01;
      double u02;
      double q;

      // #pragma omp for
      for (i = 1; i < m_N-1; i++){
        for (j = 1; j < m_N-1; j++){
          // Extract elements

          // #pragma omp atomic read
          u10 = old(i+1,j);
          // #pragma omp atomic read
          u20 = old(i-1,j);
          // #pragma omp atomic read
          u01 = old(i,j+1);
          // #pragma omp atomic read
          u02 = old(i,j-1);

          q = m_q(i,j);

          // Find next approximation to solution
          m_u(i,j) = m_diag_element*(m_alpha*(m_Ax*(u10 + u20)
          + m_Ay*(u02 + u01)) + q);
        }
      }

      // Check convergence, loop exits if s is less than the tolerance specified
      s = 0;
      double term = 0;

      // #pragma omp for
      for (i = 0; i < m_N; i++){
        for (j = 0; j < m_N; j++){
          term = old(i,j) - m_u(i,j);
          s += term*term;

          // Overwrite old solution
          old(i,j) = m_u(i,j);
        }
      }
    }
    k++;
  }
  if (k==m_maxiter){
    // Output error/warning if solution did not converge within set number of max iterations
    std::cerr << "Solution using Jacobi iterative method did not converge properly within set limit of maximum iterations." << std::endl;
    std::cout << "Final sum: " << s << std::endl;
  }
}

void DiffusionEquationSolver2D::solve(){
  // Set initial condition
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_u(i,j) = m_init_func(i*m_h,j*m_h);
    }
  }

  // Write initial state to file
  write_to_file();

  double wtime = omp_get_wtime();
  // Time iteration
  for (m_t = 1; m_t <= m_M; m_t++){
    // Set source term
    set_source_term();

    // Move one step in time
    jacobi();

    // Write to file
    if (m_t%m_write_limit == 0){
      write_to_file();
    }

    // Write errors to file
    if (m_write_errors){
      calculate_and_output_errors();
    }
  }
  wtime = omp_get_wtime() - wtime;  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of output file
  std::cout << "Finished simulating." << "\nElapsed time in seconds = " << wtime << std::endl;
}

void DiffusionEquationSolver2D::write_to_file(){
  // Write flattened matrix to file
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_ofile << std::setw(15) << std::setprecision(8) << m_u(i,j) << ' ';
    }
  }
  // Write time to same line
  m_ofile << std::setw(15) << std::setprecision(8) << m_t*m_dt << std::endl;
}


void DiffusionEquationSolver2D::set_source_term(){
  // Calculate source term used in jacobi solver
  if (m_use_source_term){
    // Use source term function if specified and previous timestep
    for (int i = 0; i < m_N; i++){
      for (int j = 0; j < m_N; j++){
        m_q(i,j) = m_dt*m_source_term(i*m_h,j*m_h,m_t*m_dt) + m_u(i,j);
      }
    }
  } else {
    // Use only the previous timestep if no source function is specified
    m_q = m_u;
  }
}

void DiffusionEquationSolver2D::compare_with_analytic(double (*analytic)(double, double, double), std::string error_filename){
  m_analytic = analytic;                                            // Function that returns analytic solution
  m_u_analytic = arma::zeros<arma::mat>(m_N,m_N);                   // Matrix to store analytic solution
  m_error_filename = error_filename;                                // Name of error output file
  m_error_ofile.open(m_error_filename.c_str(), std::ofstream::out); // Ofstream object of error output file
  m_write_errors = true;                                            // Bool used to tell the solver to write errors
}

void DiffusionEquationSolver2D::calculate_and_output_errors(){
  // Calculate analytic solution
  for (int i = 0; i <= m_N; i++){
    for (int j = 0; j <= m_N; j++){
      m_u_analytic(i,j) = m_analytic(i*m_h,j*m_h,m_t*m_dt);
    }
  }

  // Get absolute error matrix
  arma::mat diff = m_u - m_u_analytic;
  double err = arma::accu(arma::sqrt(diff*diff))/arma::accu(m_u_analytic*m_u_analytic);

  // Write error to file
  m_error_ofile << std::setw(15) << std::setprecision(8) << err << ' ';
  // Also write the current time to file
  m_error_ofile << std::setw(15) << std::setprecision(8) << m_t*m_dt << std::endl;
}
