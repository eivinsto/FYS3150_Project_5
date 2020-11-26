#include "tridiag_solver.hpp"
#include <iostream>
#include <armadillo>
#include <cmath>

using namespace arma;

TriDiag::TriDiag(int N, double h, vec& u_anal, vec& x){
  m_N = N;
  m_h = h;
  m_h2 = h*h;
  m_u_anal = u_anal;
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

void TriDiag::find_relative_error()
{
  /* Calculates log10 of relative error in all steps, and returns
  *  the maximum of these values.
  */
  vec epsilon = zeros<vec>(m_N);
  for (int i=1; i<m_N-1; ++i){
    epsilon[i] = log10( abs((m_u[i]-m_u_anal[i]) / m_u_anal[i]) );

  }
  m_eps = max(epsilon);
}

void TriDiag::solve() {
  m_u = zeros<vec>(m_N);          // Vector for numerical solution
  m_b_twiddle = zeros<vec>(m_N);  // Vector with known points.
  m_b_recip = zeros<vec>(m_N);    // Vector with 1/b.

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

  // setting known values:
  m_u[0] = 0;
  m_u[m_N-1] = 0;

  // Calculate maximum of log10 of relative error
  find_relative_error();
  cout << "Maximum (log10 of) relative error in special algorithm with " << m_N
       << " steps: " << m_eps << endl;
}

void TriDiag::save_data(std::string filename){
  m_u.save(filename, raw_binary);
  // m_u.save("u_special" + std::to_string(m_N) + ".txt", arma_ascii);
}
