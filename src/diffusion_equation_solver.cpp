#include "diffusion_equation_solver.hpp"
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <cmath>
#include <string>

DiffusionEquationSolver::DiffusionEquationSolver(int N, double dt, double(*init_func)(double), std::string method, std::string filename){
  m_N = N;
  m_dt = dt;
  m_dx = 1/(N+1);
  m_u = arma::zeros(m_N+1);
  m_y = arma::zeros(m_N+1);
  m_init_func = init_func;
  m_alpha = m_dt/(m_dx*m_dx);
  m_a = m_c = -m_alpha;
  m_method = method;
  m_output_filename = filename;
  m_ofile.open(m_output_filename.c_str(), std::ofstream::out);

  // If-else block to determine value of m_b pertaining to the choice of method
  if (m_method=="ForwardEuler"){
    m_coeff = 1-2*m_alpha;
  } else if (m_method=="BackwardEuler"){
    m_b = 1 + 2*m_alpha;
  } else if (m_method=="CrankNicholson"){
    m_b = 2 + 2*m_alpha;
    m_coeff = 2 - 2*m_alpha;
  } else {
    std::cout << "Invalid method specified. When initiating DiffusionEquationSolver class, please specify one of the following allowed methods:" << std::endl;
    std::cout << "ForwardEuler" << std::endl;
    std::cout << "BackwardEuler" << std::endl;
    std::cout << "CrankNicholson" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "Your input: " << m_method << std::endl;
    std::exit(0);
  }

}

void DiffusionEquationSolver::tridiag(){
  double decomp_factor = m_a/m_b;
  double b_temp = m_b - m_c*decomp_factor;
  arma::vec b_twiddle = m_y;

  for (int i = 1; i < m_N+1; i++){
    b_twiddle(i) -= b_twiddle(i-1)*decomp_factor;
  }

  m_u(m_N) = b_twiddle(m_N)/b_temp;

  for (int i = m_N; i >= 1; i--){
    m_u(i-1) = (b_twiddle(i-1) - m_c*m_u(i))/b_temp;
  }
}

// M amount of timesteps
void DiffusionEquationSolver::forward_euler_solve(){
  // Set boundary conditions
  m_u(0) = m_u(m_N) = 0;

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_y(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= m_M; j++){
    for (int i = 1; i < m_N; i++){
      m_u(i) = m_coeff*m_y(i) + m_alpha*(m_y(i+1) + m_y(i-1));
    }

    // Update previous solution
    m_y = m_u;

    // Write to file
    if (j%m_write_limit==0){
      write_to_file();
    }
  }
}

// M amount of timesteps
void DiffusionEquationSolver::backward_euler_solve(){
  // Set boundary conditions
  m_u(0) = m_u(m_N) = 0;

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_u(i) = m_y(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= m_M; j++){
    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_u(m_N) = 0;

    // Update previous solution
    m_y = m_u;

    // Write to file
    if (j%m_write_limit==0){
      write_to_file();
    }
  }
}

void DiffusionEquationSolver::crank_nicholson_solve(){
  // Set initial condition
  for (int i = 1; i < m_N; i++){
    m_u(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= m_M; j++){
    // Set correct y
    for (int i = 1; i <= m_N; i++){
      m_y(i) = m_coeff*m_u(i) + m_alpha*(m_u(i-1) + m_u(i+1));
    }
    m_y(0) = m_y(m_N) = 0;

    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_u(m_N) = 0;

    // Write to file
    if (j%m_write_limit==0){
      write_to_file();
    }
  }
}

void DiffusionEquationSolver::solve(int M, int write_limit){
  m_M = M;
  m_write_limit = write_limit;
  if (m_method=="ForwardEuler"){ forward_euler_solve(); }
  else if (m_method=="BackwardEuler"){ backward_euler_solve(); }
  else if (m_method=="CrankNicholson"){ crank_nicholson_solve(); }
  else {
    std::cout << "Method specification does not match any implemented methods. Error occured during call to solve()." << std::endl;
    std::exit(0);
  }
}

void DiffusionEquationSolver::write_to_file(){
  for (int i = 0; i<=m_N; i++){
    m_ofile << std::setw(15) << std::setprecision(8) << m_u(i) << ' ';
  }
  m_ofile << std::endl;
}
