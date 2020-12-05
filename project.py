from subprocess import run, Popen, PIPE
import multiprocessing as mp
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"
datadir = rootdir + "/data/"


def build_cpp():
    """Function building cpp program."""
    run(["make", "all"], cwd=src)


def run_1D(filename, method, N, dt, M, u_b, l_b, write_limit):
    build_cpp()
    run(["./main.exe", "1D", f"{N}", f"{dt}", f"{M}", f"{write_limit}", method,
         filename, f"{u_b}", f"{l_b}"], cwd=src)


def run_2D(filename, N, dt, M, write_limit):
    build_cpp()
    run(["./main.exe", "2D", f"{N}", f"{dt}", f"{M}", f"{write_limit}",
         filename], cwd=src)


def import_data_2D(file, tsteps, N):
    data = np.genfromtxt(file)
    reformatted_data = np.zeros([tsteps, N, N])
    t = np.zeros(tsteps)
    for k in range(tsteps):
        for ix in range(N):
            reformatted_data[k, :, ix] = data[k, ix*N:(ix+1)*N]
        t[k] = data[k, -1]

    return t, reformatted_data


runflag = "start"
if __name__ == "__main__":
    print("Runs: 1D, 2D.")
    while (runflag != "1d" and runflag != "2d"):
        runflag = input("Choose run: ").lower()
        if runflag == "q" or runflag == "quit":
            sys.exit(0)

    genflag = input("Generate data? y/n: ").lower()

    # 1D sample run
    if runflag == "1d":
        Ns = [10, 100]
        dt = [0.5*0.5*1/(N)**2 for N in Ns]
        T = 100
        M = [int(T/dt[j] - 1) for j in range(len(Ns))]
        write_limit = 100
        methods = ["ForwardEuler", "BackwardEuler", "CrankNicholson"]

        output_files = []
        for method in methods:
            output_files.append([datadir + method + f"_{N}.dat" for N in Ns])

        u_b = 1
        l_b = 0

        if genflag == "y":
            for i, method in enumerate(methods):
                for j, N in enumerate(Ns):
                    run_1D(output_files[i][j], method, N, dt[j], M[j], u_b,
                           l_b, write_limit)

        data = {}
        for i, method in enumerate(methods):
            for j, N in enumerate(Ns):
                data[method, N] = np.genfromtxt(output_files[i][j])

        for i, method in enumerate(methods):
            f, ax = plt.subplots(1, 2)
            for j, N in enumerate(Ns):
                x = np.linspace(0, 1, N+1)
                ax[j].plot(x, data[method, N][0, :],
                           label=f"$t_{1} = $ {dt[j]*0}")
                ax[j].plot(x, data[method, N][-1, :],
                           label=f"$t_{2} = $ {dt[j]*(M[j]+1)}")
                ax[j].set_title(f"dx = {1/N}")
                ax[j].legend()
                ax[j].grid()

            f.suptitle(method)

        plt.show()

    # 2D sample run
    if runflag == "2d":
        N = 100
        dt = 1e-4
        M = 10000
        write_limit = 1000
        output_filename = datadir + "test2d.dat"

        if genflag == "y":
            run_2D(output_filename, N, dt, M, write_limit)

        tsteps = int(M/write_limit)
        t, data = import_data_2D(output_filename, tsteps, N)

        f, (ax1, ax2) = plt.subplots(1, 2)
        c1 = ax1.imshow(data[0, :, :], interpolation='none',
                        origin="lower", aspect='auto', extent=[0, 1, 0, 1])
        ax1.set_title(f"t = {t[0]}")
        ax1.grid()
        c2 = ax2.imshow(data[-1, :, :], interpolation='none',
                        origin="lower", aspect='auto', extent=[0, 1, 0, 1])
        ax2.set_title(f"t = {t[-1]}")
        ax2.grid()
        f.colorbar(c1, ax=ax1)
        f.colorbar(c2, ax=ax2)
        plt.show()
