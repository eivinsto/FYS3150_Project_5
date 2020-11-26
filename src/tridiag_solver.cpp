#include "tridiag_solver.hpp"
#include <iostream>
#include <armadillo>
#include <cmath>

TriDiag::TriDiag(int N, double h, arma::vec& x){
  m_N = N;
  m_h = h;
  m_h2 = h*h;
  m_x = x;
}

void TriDiag::special_forward()
{
  //forward loop
  for (int i = 1; i<m_N; ++i){
    m_b_twiddle[i] = m_b_twiddle[i] + m_b_twiddle[i-1]*m_b_recip[i-1];
  }
}

void TriDiag::special_backward()
{
  //setting first element
  m_u[m_N-1] = m_b_twiddle[m_N-1]/m_b_recip[m_N-1];

  //backward loop
  for (int i = m_N-1; i>=1; --i){
    m_u[i-1] = (m_b_twiddle[i-1] + m_u[i])*m_b_recip[i-1];
  }
}

double TriDiag::relative_error(arma::vec& u_anal)
{
  /* Calculates log10 of relative error in all steps, and returns
  *  the maximum of these values.
  */
  arma::vec epsilon = arma::zeros(m_N);
  for (int i=1; i<m_N-1; ++i){
    epsilon[i] = std::log10( std::abs((m_u[i]-u_anal[i]) / u_anal[i]) );

  }
  return arma::max(epsilon);
}

void TriDiag::solve() {
  m_u = arma::zeros(m_N);          // Vector for numerical solution
  m_b_twiddle = arma::zeros(m_N);  // Vector with known points.
  m_b_recip = arma::zeros(m_N);    // Vector with 1/b.

  // Generating values for input function
  for (int i = 0; i<m_N; ++i){
    m_b_twiddle[i] = m_h2*100*exp(-10*m_x[i]);
  }

  // Calculating 1/b where b is the diagonal elements.
  for (int i = 1; i < m_N+1; ++i) {
    m_b_recip[i-1] = i/(i + 1.0);
  }

  special_forward();
  special_backward();
}

void TriDiag::save_data(std::string filename){
  m_u.save(filename, arma::raw_binary);
  // m_u.save("u_special" + std::to_string(m_N) + ".txt", arma_ascii);
}
