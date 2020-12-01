#include <armadillo>
#include <iostream>
#include <string>
#include <fstream>

class DiffusionEquationSolver1D{
public:
  DiffusionEquationSolver1D(int, double, int, int, double(*)(double), std::string,
                          std::string, double, double);
  void solve();

private:
  void forward_euler_solve();
  void backward_euler_solve();
  void crank_nicholson_solve();
  void tridiag();
  double (*m_init_func)(double);
  void write_to_file();

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
