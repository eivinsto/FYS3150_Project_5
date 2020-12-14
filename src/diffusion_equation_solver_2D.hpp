#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>

/**
* Class that is used to solve the diffusion equation in two dimensions using
* an implicit scheme and the Jacobi iterative method. The class is set up such
* that it can also handle a source term in the equation.
*/
class DiffusionEquationSolver2D{
public:
  // Constructors
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string);
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string, double(*)(double, double, double),
                            double, double);
  // Public methods
  void solve();
  void compare_with_analytic(double (*)(double,double,double), std::string);
  void new_source_term(double (*)(double, double, double), std::string, int, int);

private:
  // Private methods
  void jacobi();
  void write_to_file();
  void set_source_term();
  void calculate_and_output_errors();
  double (*m_analytic)(double, double, double);
  double (*m_y_ub)(double);
  double (*m_y_lb)(double);
  double (*m_x_ub)(double);
  double (*m_x_lb)(double);
  double (*m_init_func)(double, double);
  double (*m_source_term)(double, double, double);

  // Private variables
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
  double m_abstol = 1e-10;
  int m_write_limit;
  double m_diag_element;
  bool m_use_source_term = false;
  double m_Ax = 1;
  double m_Ay = 1;
  arma::mat m_u_analytic;
  std::string m_error_filename;
  std::ofstream m_error_ofile;
  bool m_write_errors = false;
  bool m_initialized = false;
};
