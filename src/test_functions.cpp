#include "catch.hpp"
#include "diffusion_equation_solver_1D.hpp"
#include "diffusion_equation_solver_2D.hpp"
#include <string>
#include <fstream>

bool file_exists(std::string filename) {
  std::ifstream infile(filename);
  return infile.good();
}

double init_func(double x){
  return 0;
}

double init_func2D(double x, double y){
  return 0;
}

double x_ub2D(double y){
  return 0;
}

double x_lb2D(double y){
  return 0;
}

double y_ub2D(double x){
  return 0;
}

double y_lb2D(double x){
  return 0;
}


TEST_CASE("Test file creation on DiffusionEquationSolver1D constructor call.") {
  std::string filename = "../data/1D-filetest.dat";
  int N = 2;
  int M = 1;
  double dt = 0.49*0.5*0.5;
  int write_limit = 1;
  std::string method = "ForwardEuler";

  double u_b = 1;
  double l_b = 0;

  REQUIRE(!file_exists(filename));
  {
    DiffusionEquationSolver1D system(M, dt, N, write_limit, init_func, method, filename, u_b, l_b);
  }
  REQUIRE(file_exists(filename));
  std::remove(filename.c_str());
}


TEST_CASE("Test file creation on DiffusionEquationSolver2D constructor call.") {
  std::string filename = "../data/single-filetest.dat";

  int N = 4;
  int M = 1;
  double dt = 0.49*0.5*0.5;
  int write_limit = 1;

  REQUIRE(!file_exists(filename));
  {
    DiffusionEquationSolver2D system(N, dt, M, write_limit, init_func2D, y_ub2D, y_lb2D, x_ub2D, x_lb2D, filename);
  }
  REQUIRE(file_exists(filename));
  std::remove(filename.c_str());
}


TEST_CASE("Test DiffusionEquationSolver1D ForwardEuler solver.") {
  std::string filename = "../data/1D-forward-filetest.dat";
  int N = 2;
  int M = 1;
  double dx_square = 0.5*0.5;
  double dt = 0.49*dx_square;
  int write_limit = 1;
  std::string method = "ForwardEuler";

  double u_2_0 = init_func(0.5);

  double u_b = 1;
  double l_b = 0;
  double u_2_1 = u_2_0 + (dt/dx_square)*(u_b - 2*u_2_0 + l_b);

  DiffusionEquationSolver1D system(N, dt, M, write_limit, init_func, method, filename, u_b, l_b);
  system.solve();

  arma::mat result;
  result.load(filename, arma::auto_detect);
  std::remove(filename.c_str());
  REQUIRE(int(result.n_cols)-2 == N);
  REQUIRE(int(result.n_rows)-1 == M);

  REQUIRE(result(1, 1) == Approx(u_2_1));

}


TEST_CASE("Test DiffusionEquationSolver1D BackwardEuler solver.") {
  std::string filename = "../data/1D-backward-filetest.dat";
  int N = 4;
  int M = 100;
  double dx_square = 0.5*0.5;
  double dt = 0.49*dx_square;
  int write_limit = 1;
  std::string method = "BackwardEuler";

  double u_b = 0;
  double l_b = 0;

  DiffusionEquationSolver1D system(N, dt, M, write_limit, init_func, method, filename, u_b, l_b);
  system.solve();

  arma::mat result;
  result.load(filename, arma::auto_detect);
  std::remove(filename.c_str());

  for (int i = 0; i < M+1; ++i) {
    for (int j = 0; j < N+1; ++j) {
      REQUIRE(result(i, j) == Approx(0.0).margin(1e-12));
    }
  }
}


TEST_CASE("Test DiffusionEquationSolver2D CrankNicholson solver.") {
  std::string filename = "../data/2D-CN-filetest.dat";
  int N = 10;
  int M = 100;
  double dx_square = 0.5*0.5;
  double dt = 0.49*dx_square;
  int write_limit = 1;

  DiffusionEquationSolver2D system(N, dt, M, write_limit, init_func2D, y_ub2D, y_lb2D, x_ub2D, x_lb2D, filename);
  system.solve();

  arma::mat result;
  result.load(filename, arma::auto_detect);
  std::remove(filename.c_str());

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < (N+1)*(N+1); ++j) {
      REQUIRE(result(i, j) == Approx(0.0).margin(1e-12));
    }
  }

  for (int i = 0; i < M; ++i) {
    REQUIRE(result(i, (N+1)*(N+1)) == Approx(dt*i));
  }
}
