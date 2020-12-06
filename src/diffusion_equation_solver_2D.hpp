#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>

class DiffusionEquationSolver2D{
public:
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string);
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string, double(*)(double, double, double));
  void solve();

private:
  void jacobi();
  void write_to_file();
  void set_source_term();
  double (*m_y_ub)(double);
  double (*m_y_lb)(double);
  double (*m_x_ub)(double);
  double (*m_x_lb)(double);
  double (*m_init_func)(double, double);
  double (*m_source_term)(double, double, double);

  int m_t = 0;
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
  double m_alpha_coeff;
  bool m_use_source_term = false;

};
