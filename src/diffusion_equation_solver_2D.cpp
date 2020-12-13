#include <string>
#include <iostream>
#include <iomanip>
#include "diffusion_equation_solver_2D.hpp"
#include <fstream>
#include <armadillo>
#include <cmath>
#include <omp.h>


/**
* First constructor for the DiffusionEquationSolver2D class. This constructor
* assigns the necessary member variables of the class.
*
* N - integer, amount of steps in length
* dt - double, timestep
* M - integer, amount of time steps
* write_limit - integer, write results every <write_limit> timesteps
* init_func - function pointer, initial condition of the system (should return correct
*             initial state u as u = init_func(x,y), where x and y are the
*             length coordinates)
* y_ub - function pointer, should return upper boundary condition in y-direction
*        u(x,1) = y_ub(x)
* y_lb - function pointer, should return lower boundary condition in y-direction
*        u(x,0) = y_lb(x)
* x_ub - function pointer, should return upper boundary condition in y-direction
*        u(1,y) = x_ub(y)
* x_lb - function pointer, should return lower boundary condition in y-direction
*        u(0,y) = x_lb(y)
* ofilename - string, name of file to write results to
*/
DiffusionEquationSolver2D::DiffusionEquationSolver2D(int N, double dt, int M, int write_limit,
                                                     double (*init_func)(double, double),
                                                     double (*y_ub)(double), double (*y_lb)(double),
                                                     double (*x_ub)(double), double (*x_lb)(double),
                                                     std::string ofilename){

  m_init_func = init_func;                               // Initial condition function
  m_y_ub = y_ub;                                         // Upper boundary function in y-direction
  m_y_lb = y_lb;                                         // Lower boundary function in y-direction
  m_x_ub = x_ub;                                         // Upper boundary function in x-direction
  m_x_lb = x_lb;                                         // Lower boundary function in x-direction
  m_N = N+1;                                             // Amount of lengthsteps
  m_dt = dt;                                             // Timestep
  m_M = M;                                               // Amount of timesteps
  m_ofilename = ofilename;                               // Output filename
  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of output file
  m_h = 1.0/double(N);                                   // Lengthstep
  m_alpha = m_dt/(m_h*m_h);                              // Coefficient for use in time integration
  m_u = arma::zeros<arma::mat>(m_N,m_N);                 // Solution matrix
  m_q = arma::zeros<arma::mat>(m_N,m_N);                 // Source term
  m_write_limit = write_limit;                           // Write every <write_limit> timesteps
  m_diag_element = 1.0/(1.0+4.0*m_alpha);
}


/**
* Second constructor for the DiffusionEquationSolver2D class. This constructor
* takes all the same inputs as the first constructor and a few additional ones.
* This constructor is to be used if a source term should be added.
* This constructor also calls the first constructor with the input variables
* that are the same: N, dt, M, write_limit, init_func, y_ub, y_lb, x_ub, x_lb
* and ofilename. See documentation of first constructor for details on these.
*
* source_term - function pointer, should return source term of diffusion equation
*               q(x,y,t) = source_term(x,y,t)
* ax - Parameter used to simulate rectangular systems with a square grid.
*      See pdf in /doc/-folder of repo for further details on this.
* ay - Parameter used to simulate rectangular systems with a square grid.
*      See pdf in /doc/-folder of repo for further details on this.
*/
DiffusionEquationSolver2D::DiffusionEquationSolver2D(int N, double dt, int M, int write_limit,
                                                     double (*init_func)(double, double),
                                                     double (*y_ub)(double), double (*y_lb)(double),
                                                     double (*x_ub)(double), double (*x_lb)(double),
                                                     std::string ofilename, double (*source_term)(double,double,double),
                                                     double ax, double ay)
                         : DiffusionEquationSolver2D(N, dt, M, write_limit, init_func, y_ub, y_lb,
                                                     x_ub, x_lb, ofilename){
  m_use_source_term = true;                             // Specify that source term function has been provided
  m_source_term = source_term;                          // Store source term function
  m_Ax = 1.0/(ax*ax);                                   // Calculate constant for x-direction used for rectangular grid
  m_Ay = 1.0/(ay*ay);                                   // Calculate constant for y-direction used for rectangular grid
  m_diag_element = 1.0/(1.0 + 2*m_alpha*(m_Ax + m_Ay)); // Specify diagonal elements in terms of Ax and Ay
}

/**
* This function performs the jacobi iterative method used to solve Au = q, where
* the A is a tensor specialized to the diffusion equation, and u and q are matrices.
* The solution in the previous time step and any source terms are included in
* the "constant" matrix q (m_q), and the solution that is returned is stored
* in the matrix m_u.
*/
void DiffusionEquationSolver2D::jacobi(){
  // Define sum variable
  double s = 1;

  // Boundary conditions
  for (int i = 0; i < m_N; i++){
    m_u(0,i) = m_x_lb(i*m_h);
    m_u(m_N-1,i) = m_x_ub(i*m_h);
    m_u(i,0) = m_y_lb(i*m_h);
    m_u(i,m_N-1) = m_y_ub(i*m_h);
  }

  // Generate dense matrix to store previous solution
  arma::mat old = m_u; // Initial guess is value in previous timestep

  int k = 0;
  int i, j;
  // int thrds = 0.5*omp_get_max_threads();
  // Iterative solver
  while (std::sqrt(s) > m_abstol && k < m_maxiter){
    // #pragma omp parallel if(m_N > 1000) num_threads(2) default(shared) private(i, j) firstprivate(m_diag_element, m_alpha, m_Ax, m_Ay) reduction(+:s)
    {
      // Define elements used for extraction
      double u10;
      double u20;
      double u01;
      double u02;
      double q;

      // #pragma omp for
      for (i = 1; i < m_N-1; i++){
        for (j = 1; j < m_N-1; j++){
          // Extract elements

          // #pragma omp atomic read
          u10 = old(i+1,j);
          // #pragma omp atomic read
          u20 = old(i-1,j);
          // #pragma omp atomic read
          u01 = old(i,j+1);
          // #pragma omp atomic read
          u02 = old(i,j-1);

          q = m_q(i,j);

          // Find next approximation to solution
          m_u(i,j) = m_diag_element*(m_alpha*(m_Ax*(u10 + u20)
          + m_Ay*(u02 + u01)) + q);
        }
      }

      // Check convergence, loop exits if sqrt(s) is less than the tolerance
      // specified in member variable m_abstol (defined in header)
      s = 0;
      double term = 0;

      // #pragma omp for
      for (i = 0; i < m_N; i++){
        for (j = 0; j < m_N; j++){
          term = old(i,j) - m_u(i,j);
          s += term*term;

          // Overwrite old solution
          old(i,j) = m_u(i,j);
        }
      }
    }
    k++;
  }
  if (k==m_maxiter){
    // Output error/warning if solution did not converge within set number of max iterations
    std::cerr << "Solution using Jacobi iterative method did not converge properly within set limit of maximum iterations." << std::endl;
    std::cout << "Final sum: " << std::sqrt(s) << std::endl;
  }
}


/**
* Method that solves the system for the specified amount of timesteps. This
* method generates the initial state, and performs jacobi iterations using
* the jacobi() method to move the system ahead in time.
*/
void DiffusionEquationSolver2D::solve(){
  // Time implementation
  double wtime = omp_get_wtime();
  // Set initial condition
  if (!m_initialized){
    for (int i = 0; i < m_N; i++){
      for (int j = 0; j < m_N; j++){
        m_u(i,j) = m_init_func(i*m_h,j*m_h);
      }
    }
    // Do not reinitialize if solve is called a second time (for use with
    // new_source_term() method).)
    m_initialized = true;
  }

  // Write initial state and errors (if specified) to file
  write_to_file();
  if (m_write_errors){
    calculate_and_output_errors();
  }

  // Time iteration
  for (m_t = 1; m_t <= m_M; m_t++){
    // Set source term
    set_source_term();

    // Move one step in time
    jacobi();

    // Write to file
    if (m_t%m_write_limit == 0){
      write_to_file();
    }

    // Write errors to file
    if (m_write_errors){
      calculate_and_output_errors();
    }
  }
  // Finish timing implementation and print results
  wtime = omp_get_wtime() - wtime;
  std::cout << "Finished simulating." << "\nElapsed time in seconds = " << wtime << std::endl;
}

/**
* Method that writes results to file. This writes a flattened version of the
* matrix m_u for each line in the output file, and adds the current time in the
* simulation to the end of each line.
*/
void DiffusionEquationSolver2D::write_to_file(){
  // Write flattened matrix to file
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_ofile << std::setw(15) << std::setprecision(8) << m_u(i,j) << ' ';
    }
  }
  // Write time to same line
  m_ofile << std::setw(15) << std::setprecision(8) << m_t*m_dt << std::endl;
}

/**
* This function calculates the matrix m_q. If no source term is specified, this
* simply sets it to the value in the previous timestep. If a source term is
* specified this is added as well.
*/
void DiffusionEquationSolver2D::set_source_term(){
  // Calculate source term used in jacobi solver
  if (m_use_source_term){
    // Use source term function if specified and previous timestep
    for (int i = 0; i < m_N; i++){
      for (int j = 0; j < m_N; j++){
        m_q(i,j) = m_dt*m_source_term(i*m_h,j*m_h,m_t*m_dt) + m_u(i,j);
      }
    }
  } else {
    // Use only the previous timestep if no source function is specified
    m_q = m_u;
  }
}

/**
* This function is used to change the source term during a simulation.
* This is intended to be used in the following way:
* - Set up a simulation with constructor and run said simulation
* - Specify new source term using this method, and also specify amount of timesteps
*   M, write_limit, and name of output file anew.
* - Run a second simulation with the new source term using the final state of the
*   previous simulation as the inital state of the second one.
* Using this method can be used to run an indefinite amount of chained simulations.
* Simulation time is also reset between simulations, so if a shared simulation time
* is desired, the simulation times must be added manually in processing of the data.
*
* source_term - function pointer, new source term function to be used in secon simulation
* ofilename - string, name of new output file for results
* M_new - integer, amount of timesteps for second simulation
* write_limit_new - integer, new limit on how often to write results to output file
*/
void DiffusionEquationSolver2D::new_source_term(double (*source_term)(double, double, double),std::string ofilename, int M_new, int write_limit_new){
  m_t = 0;                                               // Reset simulation time
  m_source_term = source_term;                           // Set new source term function
  m_ofile.close();                                       // Close old output file
  m_ofilename = ofilename;                               // Set new output file name
  m_ofile.open(m_ofilename.c_str(), std::ofstream::out); // Ofstream object of new output file
  m_M = M_new;                                           // Overwrite amount of timesteps
  m_write_limit = write_limit_new;                       // Overwrite write limit
}

/**
* Method to be called in order to perform comparison with an analytic solution.
* This method takes a function that returns an analytic solution, along with a
* filename to output error values to, and stores them as member variables of the
* class.
*
* analytic - function pointer, should return analytic solution when given arguments
*            (x,y,t)
* error_filename - string, name of file to store output error values in
*/
void DiffusionEquationSolver2D::compare_with_analytic(double (*analytic)(double, double, double), std::string error_filename){
  m_analytic = analytic;                                            // Function that returns analytic solution
  m_u_analytic = arma::zeros<arma::mat>(m_N,m_N);                   // Matrix to store analytic solution
  m_error_filename = error_filename;                                // Name of error output file
  m_error_ofile.open(m_error_filename.c_str(), std::ofstream::out); // Ofstream object of error output file
  m_write_errors = true;                                            // Bool used to tell the solver to write errors
}

/**
* Method that calculates the relative RMS error of the system (RMS of absolute
* difference matrix divided by RMS of analytic solution), and writes the results
* to an output file, along with the simulation time at which the error
* measurement was made.
*/
void DiffusionEquationSolver2D::calculate_and_output_errors(){
  // Calculate analytic solution
  for (int i = 0; i < m_N; i++){
    for (int j = 0; j < m_N; j++){
      m_u_analytic(i,j) = m_analytic(i*m_h,j*m_h,m_t*m_dt);
    }
  }

  // Get absolute error matrix
  arma::mat diff = m_u - m_u_analytic;

  // Calculate relative RMS error 
  double err = sqrt(arma::accu(diff%diff)/arma::accu(m_u_analytic%m_u_analytic));

  // Write error to file
  m_error_ofile << std::setw(15) << std::setprecision(8) << err << ' ';
  // Also write the current time to file
  m_error_ofile << std::setw(15) << std::setprecision(8) << m_t*m_dt << std::endl;
}
