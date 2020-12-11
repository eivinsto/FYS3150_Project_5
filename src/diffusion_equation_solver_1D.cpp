#include "diffusion_equation_solver_1D.hpp"
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <cmath>
#include <string>
#include <fstream>
#include <omp.h>

/**
* Constructor for the DiffusionEquationSolver1D class. This constructor assigns
* the necessary member variables of the class.
*
* N - integer, amount of steps in length
* dt - double, timestep
* M - integer, amount of time steps
* write_limit - integer, write results every <write_limit> timesteps
* init_func - function, initial condition of the system (should return correct
*             initial state u as u = init_func(x), where x is the length coordinate)
* method - string, should contain the method of choice, can be either:
*          ForwardEuler, BackwardEuler, or CrankNicolson.
* filename - string, name of file to write results to
* u_b - double, lower boundary value
* l_b - lower boundary value
*/
DiffusionEquationSolver1D::DiffusionEquationSolver1D(int N, double dt, int M, int write_limit,
                                                 double(*init_func)(double), std::string method,
                                                 std::string filename, double u_b, double l_b){
  m_N = N;                      // Amount of lengthsteps
  m_dt = dt;                    // Timestep
  m_dx = 1.0/double(N);         // Lengthstep
  m_M = M;                      // Amount of timesteps
  m_write_limit = write_limit;  // Write results every <write_limit> timesteps
  m_u = arma::zeros(m_N+1);     // Vector containing solution
  m_y = arma::zeros(m_N+1);     // Vector containing solution in previous timestep
  m_init_func = init_func;      // Initial condition function
  m_alpha = m_dt/(m_dx*m_dx);   // Coefficient for use in all algorithms
  m_a = m_c = -m_alpha;         // Lower and upper diagonal elements of tridiagonal matrix
  m_method = method;            // Which scheme to use when solving the system
  m_output_filename = filename; // Name of file which results should be written to (output file)
  m_ofile.open(m_output_filename.c_str(), std::ofstream::out);  // Ofstream object of output file
  m_ub = u_b;                   // Upper boundary value
  m_lb = l_b;                   // Lower boundary value

  // If-else block to determine value of m_b pertaining to the choice of method
  if (m_method=="ForwardEuler"){
    m_coeff = 1.0-2.0*m_alpha;  // Precalculated coefficient for use in moving system in time
    // Print warning if solution will not converge
    if (m_alpha >= 1.0/2){
      std::cout << "Warning: Soution will not converge properly with this choice of parameters N and dt." << std::endl;
      std::cout << "They need to be such that dt/dx^2 < 1/2, where dx = 1/(N+1). Currently dt/dx^2 = " << m_alpha << std::endl;
    }
  } else if (m_method=="BackwardEuler"){
    m_b = 1.0 + 2.0*m_alpha;    // Diagonal elements of matrix
  } else if (m_method=="CrankNicolson"){
    m_b = 2.0 + 2.0*m_alpha;    // Diagonal elements of matrix
    m_coeff = 2.0 - 2.0*m_alpha;// Precalculated coefficient for use in moving system in time
  } else {
    // Exit program if invalid method is specified
    std::cerr << "Invalid method specified. When initiating DiffusionEquationSolver1D class, please specify one of the following allowed methods:" << std::endl;
    std::cerr << "ForwardEuler" << std::endl;
    std::cerr << "BackwardEuler" << std::endl;
    std::cerr << "CrankNicolson" << std::endl;
    std::cerr << "--------" << std::endl;
    std::cerr << "Your input: " << m_method << std::endl;
    std::exit(0);
  }

}

/**
* Solves matrix-vector equation Au = y for u, when A is a tridiagonal matrix with
* lower diagonal elements a, diagonal elements b, and upper diagonal elements c.
* This member function takes no arguments but edits m_u so that it is moved one
* step in time.
*/
void DiffusionEquationSolver1D::tridiag(){
  // Precalculate factors to reduce necessary FLOPs
  double ac = m_a*m_c;

  // Initialize temporary vector for modified RHS of equation
  arma::vec f_tilde = m_y;
  f_tilde(1) -= m_a*m_u(0);

  // Initialize temporary vector for reduced diagonal elements
  arma::vec d_tilde = arma::zeros(m_N);
  d_tilde(1) = m_b;

  // Update RHS elements
  for (int i = 2; i < m_N; i++){
    d_tilde(i) = m_b - ac/d_tilde(i-1);
    f_tilde(i) -= f_tilde(i-1)*m_a/d_tilde(i-1);
  }

  // Find m_u (boundaries m_u(N) and m_u(0) are set manually outside of the tridiag function)
  for (int i = m_N; i >= 2; i--){
    m_u(i-1) = (f_tilde(i-1) - m_c*m_u(i))/d_tilde(i-1);
  }
}

/**
* Solves dimensionless diffusion equation in one dimension using a forward
* Euler based explicit scheme. The iterative part is a matrix multiplication where
* the matrix is tridiagonal, and with constant elements along the diagonal bands.
*/
void DiffusionEquationSolver1D::forward_euler_solve(){
  // Set boundary conditions
  m_u(0) = m_y(0) = m_lb;
  m_u(m_N) = m_y(m_N) = m_ub;

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_u(i) = m_init_func(m_dx*i);
  }

  // Write initial state to file
  write_to_file();

  // Iterate over timesteps
  for (m_j = 1; m_j <= m_M; m_j++){
    // Matrix multiplication with tridiagonal matrix
    for (int i = 1; i < m_N; i++){
      m_y(i) = m_coeff*m_u(i) + m_alpha*(m_u(i+1) + m_u(i-1));
    }

    // Update previous solution
    m_u = m_y;

    // Write to file
    if (m_j%m_write_limit==0){
      write_to_file();
    }
  }
}

/**
* Solves dimensionless diffusion equation in one dimension using a backward
* Euler based implicit scheme. This iterative part is a matrix-vector equation
* that requires the inversion of a tridiagonal matrix with constant elements
* along the diagonal bands. This calculation is performed in the tridiag() function.
*/
void DiffusionEquationSolver1D::backward_euler_solve(){

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_u(i) = m_y(i) = m_init_func(m_dx*i);
  }

  // Set boundary conditions
  m_u(0) = m_lb;
  m_u(m_N) = m_ub;

  // Write first timeste to file
  write_to_file();

  // Iterate over timesteps
  for (m_j = 1; m_j <= m_M; m_j++){
    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_lb;
    m_u(m_N) = m_ub;

    // Update previous solution
    m_y = m_u;

    // Write to file
    if (m_j%m_write_limit==0){
      write_to_file();
    }
  }
}

/**
* Solves dimensionless diffusion equation in one dimension using the Crank-CrankNicolson
* scheme. This iterative part consists of a matrix-vector multiplication and a
* matrix inversion (used to solve a matrix-vector equation). The first part is
* performed directly in this function, and the second part is performed by the
* tridiag() function.
*/
void DiffusionEquationSolver1D::crank_nicolson_solve(){
  // Set initial condition
  for (int i = 1; i < m_N; i++){
    m_u(i) = m_init_func(m_dx*i);
  }

  // Set boundary conditions
  m_u(0) = m_lb;
  m_u(m_N) = m_ub;
  write_to_file();

  // Iterate over timesteps
  for (m_j = 1; m_j <= m_M; m_j++){
    // Set correct y
    for (int i = 1; i < m_N; i++){
      m_y(i) = m_coeff*m_u(i) + m_alpha*(m_u(i-1) + m_u(i+1));
    }

    // Set boundary conditions
    m_y(0) = m_lb;
    m_y(m_N) = m_ub;

    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_lb;
    m_u(m_N) = m_ub;

    // Write to file
    if (m_j%m_write_limit==0){
      write_to_file();
    }
  }
}

/**
* Function that solves the system with the specified parameters. This calls
* the function corresponding to the scheme selected in the constructor, and is
* thus purely a wrapper function. If the specified method does not match any
* of those defined in this function, an error gets printed to the terminal
* before the program exits.
*/
void DiffusionEquationSolver1D::solve(){
  // Calls the appropriate solver for scheme specified in constructor
  double wtime = omp_get_wtime();
  if (m_method=="ForwardEuler"){ forward_euler_solve(); }
  else if (m_method=="BackwardEuler"){ backward_euler_solve(); }
  else if (m_method=="CrankNicolson"){ crank_nicolson_solve(); }
  else {
    // Output error if invalid method is specified.
    // Note that this is also done in the constructor, and is thus more
    // telling of an error in the implementation, and not on the users part.
    std::cerr << "Method specification does not match any implemented methods. Error occured during call to solve()." << std::endl;
    std::exit(0);
  }
  wtime = omp_get_wtime() - wtime;
  std::cout << "Finished simulating." << "\nElapsed time in seconds = " << wtime << std::endl;
}

/**
* Function that write solution vector to file. This is done in every timestep,
* and thus the time is not explicitly written to file.
*/
void DiffusionEquationSolver1D::write_to_file(){
  // Write components of solution vector to file
  for (int i = 0; i<=m_N; i++){
    m_ofile << std::setw(15) << std::setprecision(8) << m_u(i) << ' ';
  }
  m_ofile << std::setw(15) << std::setprecision(8) << double(m_j)*m_dt;
  m_ofile << std::endl;
}
