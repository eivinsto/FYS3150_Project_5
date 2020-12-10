#include <armadillo>
#include <iostream>
#include <string>
#include <fstream>

/**
* Class that is used to solve a dimensionless 1D diffusion equation with either
* a forward Euler based explicit scheme, a backward Euler based implicit scheme,
* or the Crank-Nicolson scheme.
*/
class DiffusionEquationSolver1D{
public:
  // Constructor
  DiffusionEquationSolver1D(int, double, int, int, double(*)(double), std::string,
                          std::string, double, double);
  // Public functions
  void solve();

private:
  // Private functions
  void forward_euler_solve();
  void backward_euler_solve();
  void crank_nicholson_solve();
  void tridiag();
  void write_to_file();
  double (*m_init_func)(double); // Function pointer used to generate initial state of system 

  // Private variables
  int m_N;
  arma::vec m_u;
  double m_dx;
  double m_dt;
  double m_alpha;
  double m_a;
  double m_b;
  double m_c;
  arma::vec m_y;
  std::string m_method;
  double m_coeff;
  std::string m_output_filename;
  std::ofstream m_ofile;
  int m_M;
  int m_write_limit;
  double m_lb;
  double m_ub;
};
