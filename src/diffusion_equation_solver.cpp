#include "diffusion_equation_solver.hpp"
#include <iostream>
#include <armadillo>
#include <cmath>

DiffusionEquationSolver::DiffusionEquationSolver(int N, double dt, double(*init_func)(double), std::string method){
  m_N = N;
  m_dt = dt;
  m_dx = 1/(N+2);
  m_u = arma::zeros(m_N+1);
  m_init_func = init_func;
  m_alpha = m_dt/(m_dx*m_dx);
  m_a = m_c = -m_alpha;

  // If-else block to determine value of m_b pertaining to the choice of method
  if (method=="ForwardEuler"){
    // This empty if block is included so that the program isn't terminated if method is set to ForwardEuler
  } else if (method=="BackwardEuler"){
    m_b = 1 + 2*m_alpha;
  } else if (method=="CrankNicholson"){
    m_b = 2 + 2*m_alpha;
  } else {
    std::cout << "Invalid method specified. When initiating DiffusionEquationSolver class, please specify one of the following allowed methods:" << std::endl;
    std::cout << "ForwardEuler" << std::endl;
    std::cout << "BackwardEuler" << std::endl;
    std::cout << "CrankNicholson" << std::endl;
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
void DiffusionEquationSolver::forward_euler_solve(int M){
  // Initialize vector to be used when updating
  m_y = arma::zeros(m_N+1);

  // Set boundary conditions
  m_u(0) = m_u(m_N) = 0;

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_y(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= M; j++){
    for (int i = 1; i < m_N; i++){
      m_u(i) = m_alpha*m_y(i-1) + (1 - 2*m_alpha)*m_y(i) + m_alpha*m_y(i+1);
    }

    // Update previous solution
    m_y = m_u;
  }
}

// M amount of timesteps
void DiffusionEquationSolver::backward_euler_solve(int M){
  // Define vector to be used when updating
  m_y = arma::zeros(m_N+1);

  // Set boundary conditions
  m_u(0) = m_u(m_N) = 0;

  // Set initial condition
  for (int i = 1; i<m_N; i++){
    m_u(i) = m_y(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= M; j++){
    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_u(m_N) = 0;

    // Update previous solution
    m_y = m_u;
  }
}

void DiffusionEquationSolver::crank_nicholson_solve(int M){
  m_y = arma::zeros(m_N+1);

  // Set initial condition
  for (int i = 1; i < m_N; i++){
    m_u(i) = m_init_func(m_dx*i);
  }

  // Iterate over timesteps
  for (int j = 1; j <= M; j++){
    // Set correct y
    for (int i = 1; i <= m_N; i++){
      m_y(i) = m_alpha*m_u(i-1) + (2 - 2*m_alpha)*m_u(i) + m_alpha*m_u(i+1);
    }
    m_y(0) = m_y(m_N) = 0;

    // Use tridiagonal solver to move one step
    tridiag();

    // Set boundary conditions
    m_u(0) = m_u(m_N) = 0;
  }
}
