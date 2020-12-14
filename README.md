# FYS3150_Project_5

### benchmark run:
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
