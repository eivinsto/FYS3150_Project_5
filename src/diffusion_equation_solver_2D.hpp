#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>

class DiffusionEquationSolver2D{
public:
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string);
  void solve();

private:
  void jacobi();
  void write_to_file();
  double (*m_y_ub)(double);
  double (*m_y_lb)(double);
  double (*m_x_ub)(double);
  double (*m_x_lb)(double);
  double (*m_init_func)(double, double);

  int m_t;
  int m_N;
  double m_dt;
  int m_M;
  std::string m_ofilename;
  std::ofstream m_ofile;
  double m_h;
  double m_alpha;
  arma::mat m_u;
  arma::mat m_q;
  double m_maxiter = 100000;
  double m_abstol = 1e-14;
  int m_write_limit;
};
