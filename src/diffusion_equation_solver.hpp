#include <armadillo>

class DiffusionEquationSolver{
public:
  DiffusionEquationSolver(int, double, double(*init_func)(double), std::string);

private:
  void forward_euler_solve(int);
  void backward_euler_solve(int);
  void crank_nicholson_solve(int);
  void tridiag();
  double (*m_init_func)(double);

  int m_N;
  arma::vec m_u;
  double m_dx;
  double m_dt;
  double m_alpha;
  double m_a;
  double m_b;
  double m_c;
  arma::vec m_y;
};
