#include <armadillo>

class TriDiag{
public:
  TriDiag(int, double, arma::vec&);
  void solve();
  void save_data(std::string);
  double relative_error(arma::vec& u_anal);

private:
  void special_forward();
  void special_backward();
  int m_N;
  double m_h;
  double m_h2;
  arma::vec m_x;
  arma::vec m_b_recip;
  arma::vec m_b_twiddle;
  arma::vec m_u;
  double m_eps;
};
