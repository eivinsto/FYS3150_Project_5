#include <iostream>
#include <string>
#include <fstream>


class DiffusionEquationSolver2D{
public:
  DiffusionEquationSolver2D(int, double, int, int, double (*)(double, double),
                            double (*)(double), double (*)(double), double (*)(double),
                            double (*)(double), std::string);
  void solve();

private:
  void jacobi();
  double (*m_y_ub)(double);
  double (*m_y_lb)(double);
  double (*m_x_ub)(double);
  double (*m_x_lb)(double);
  double (*m_init_func)(double, double);

  int m_N;
  double m_dt;
  int m_M;
  int m_write_limit;
  std::string m_ofilename;
  std::ofstream m_ofile;
};
