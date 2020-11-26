#include <armadillo>

class TriDiag{
public:
  TriDiag(int, double, arma::vec&, arma::vec&);
  void solve();

private:
  void special_forward();
  void special_backward();
  void find_relative_error();
  int m_N;
  double m_h;
  double m_h2;
  arma::vec m_u_anal;
  arma::vec m_x;
  arma::vec m_b_recip;
  arma::vec m_b_twiddle;
  arma::vec m_u;
  double m_eps;
};
