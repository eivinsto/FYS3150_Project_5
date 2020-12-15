# FYS3150_Project_5

## Usage:
When executed, the "project.py" script will ask you which part of the report to run:
*   1D  - Run simulations and data analysis of 1D solver methods, including comparison to analytic solution.
*   2D  - Run simulation and data analysis of generic 2D diffusion problem with comparison with analytic solution.
*   h / heat  - Run simulations and data analysis of heat diffusion in the lithosphere before and after radioactive enrichment.
*   b / benchmark  - Run benchmark for 1D solvers, and serial 2D solver.
*   t / test - Run unit tests for both 1D and 2D solvers.

Example:
```console
$ python project.py
Runs: 1D, 2D, [h]eat, [t]est, [b]enchmark
Choose run: 1D
Generate data? y/n: n
```
For all runs except "test", the script will ask you if you wish to generate data for the selected run.
```console
Generate data? y/n:
```
Selecting 'y' will build the C++ program, and run the selected simulation(s) and data analysis. Selecting 'n' will not run the selected simulation(s). Instead the script will attempt to load previously generated data from the /data/ directory, and perform the data analysis.

### Example run of tests:
```console
$ python project.py
Runs: 1D, 2D, [h]eat, [t]est, [b]enchmark
Choose run: t
make: Nothing to be done for 'test'.
Finished simulating.
Elapsed time in seconds = 3.429e-05
Finished simulating.
Elapsed time in seconds = 0.000378141
Finished simulating.
Elapsed time in seconds = 0.00460255
===============================================================================
All tests passed (12712 assertions in 5 test cases)
```

### Example run of benchmark:
```console
$ python project.py
Runs: 1D, 2D, [h]eat, [t]est, [b]enchmark
Choose run: b
g++ -Wall -Wextra -O3 -march=native -fopenmp -g diffusion_equation_solver_2D.cpp -c
g++ -Wall -Wextra -O3 -march=native -fopenmp -g main.o diffusion_equation_solver_1D.o diffusion_equation_solver_2D.o -o main.exe -larmadillo
Solver benchmarks. N runs.
N = 100, M = 1000, dt = 2.5e-05, u_b = 1, l_b = 0
ForwardEuler  : μ = 3.446e-04 s, σ = 2.002e-05 s
BackwardEuler : μ = 2.113e-03 s, σ = 4.144e-05 s
CrankNicolson : μ = 2.212e-03 s, σ = 2.122e-04 s
2D Solver     : μ = 7.232e-01 s, σ = 2.928e-02 s
```

## Notes on compilation:
To compile the C++ program directly, use the Makefile provided in the /src/ directory.
During compilation of 'main.cpp' and 'test_functions.cpp', the compiler may warn of unused variables 'x' and 'y' in the functions used for boundary conditions. This is to be expected, and should be ignored.
