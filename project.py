from subprocess import run, Popen, PIPE
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
    """Function for running 1D solver.

    Arguments:
    filename -- str: string containing name and absolute path to output file.
    method -- str: string containing name of solver method to use.
    N -- int: number of steps to take in space minus one.
    dt -- float: length of time step.
    M -- int: number of steps to take in time.
    u_b -- float: boundary value u(1, t) = u_b.
    l_b -- float: boundary value u(0, t) = l_b.
    write_limit -- int: number of time steps to skip when writing to file.
    """
    build_cpp()
    run(["./main.exe", "1D", f"{N}", f"{dt}", f"{M}", f"{write_limit}", method,
         filename, f"{u_b}", f"{l_b}"], cwd=src)


def run_2D(filename, N, dt, M, write_limit, errfilename=None):
    """Function for running 2D solver on generic problem.

    Arguments:
    filename -- str: string containing name and absolute path to output file.
    method -- str: string containing name of solver method to use.
    N -- int: number of steps to take in space minus one.
    dt -- float: length of time step.
    M -- int: number of steps to take in time.
    write_limit -- int: number of time steps to skip when writing to file.

    Keyword arguments:
    errfilename -- str: string containing name of, and absolute path to,
                        output file for error between numeric and
                        analytic solution (default=None).
    """
    build_cpp()
    commandlist = ["./main.exe", "2D", f"{N}", f"{dt}", f"{M}",
                   f"{write_limit}", filename, "regular"]
    if errfilename is not None:
        commandlist.append(errfilename)
    run(commandlist, cwd=src)


def run_heat(filename1, filename2, N, dt, M1, write_limit1, M2, write_limit2,
             ax, ay):
    """Function for running 2D solver on heat problem.

    Arguments:
    filename -- str: string containing name and absolute path to output file.
    method -- str: string containing name of solver method to use.
    N -- int: number of steps to take in space minus one.
    dt -- float: length of time step.
    M1 -- int: number of steps to take in time before enrichment.
    write_limit1 -- int: number of time steps to skip when writing to file
                         before enrichment.
    M1 -- int: number of steps to take in time after enrichment.
    write_limit1 -- int: number of time steps to skip when writing to file
                         after enrichment.
    ax -- float: scaling constant for x-coordinates.
    ay -- float: scaling constant for y-coordinates.
    """
    build_cpp()
    run(["./main.exe", "2D", f"{N}", f"{dt}", f"{M1}", f"{write_limit1}",
         filename1, "heat", f"{ax}", f"{ay}", f"{M2}", f"{write_limit2}",
         filename2], cwd=src)


def import_data_2D(file, tsteps, N):
    """Returns tuple with 1D-array of time values t, and 2D-array of
    u(x, y, t) values. Function reading 2D-solver data from file.

    Arguments:
    file -- str: string containing name of, and absolute path to data file.
    tsteps -- float: number of steps in time.
    N -- int: number of steps in space minus one along both axes.
    """
    t_N = N+1
    data = np.genfromtxt(file)
    reformatted_data = np.zeros([tsteps, t_N, t_N])
    t = np.zeros(tsteps)
    for k in range(tsteps):
        for ix in range(t_N):
            reformatted_data[k, :, ix] = data[k, ix*t_N:(ix+1)*t_N]
        t[k] = data[k, -1]

    return t, reformatted_data


runflag = "start"
runflags = ["1d", "2d", "h", "heat", "t", "test", "b", "benchmark"]
if __name__ == "__main__":
    print("Runs: 1D, 2D, [h]eat, [t]est, [b]enchmark")
    while (runflag not in runflags):
        runflag = input("Choose run: ").lower()
        if runflag == "q" or runflag == "quit":
            sys.exit(0)

    if runflag not in runflags[4:]:
        # Choose whether to run simulation.
        genflag = input("Generate data? y/n: ").lower()

    if runflag == "1d":
        """Perform data analysis for the 1D solvers."""
        Ns = [10, 100]  # Defined such that step length h = 1/N
        # Calculating timestep to ensure convergence:
        dts = np.asarray([0.5*0.5/(N)**2 for N in Ns])
        T = 0.4  # Amount of time to simulate
        t1 = 0.04  # first time to plot
        t2 = 0.1  # second time to plot

        # Calculating amount of timesteps to reach t1, t2, and T
        n_t1 = np.asarray(t1/dts, dtype=np.int64)
        n_t2 = np.asarray(t2/dts, dtype=np.int64)
        n_T = np.asarray(T/dts, dtype=np.int64)
        Ms = np.asarray([int(T/dt) for dt in dts])
        write_limit = 1  # write all data to file
        methods = ["ForwardEuler", "BackwardEuler", "CrankNicolson"]

        output_files = []  # creating list of filenames:
        for method in methods:
            output_files.append([datadir + method + f"_{N}.dat" for N in Ns])

        # setting boundary conditions:
        u_b = 1
        l_b = 0

        if genflag == "y":  # running simulations if needed:
            for i, method in enumerate(methods):
                for j, N in enumerate(Ns):
                    run_1D(output_files[i][j], method, N, dts[j], Ms[j], u_b,
                           l_b, write_limit)

        def anal_1d(x, t):
            """Analytic solution for 1D example
            Args:
            x -- array or float: x-value(s) to evaluate at.
            t -- array or float: t-value(s) to evaluate at.

            Returns float if x and t are float. Returns 2D or 1D array if x
            and/or t is 1D array, with first index corresponding to t,
            and second index corresponding to x.
            """
            if isinstance(t, np.ndarray):
                arr = np.empty((len(t), len(x)))
                for i in range(len(t)):
                    arr[i, :] = (np.sin(2*np.pi*x)*np.exp(-4*t[i]*np.pi**2) +
                                 x)
                return arr
            else:
                return (np.sin(2*np.pi*x)*np.exp(-4*t*np.pi**2) + x)

        # creating dictionaries for u(x, t) and t respectivly.
        data = {}
        tdict = {}
        # iterating over methods and resolutions to read data.
        for i, method in enumerate(methods):
            for j, N in enumerate(Ns):
                data[method, N] = np.genfromtxt(output_files[i][j])[:, :-1]
                tdict[method, N] = np.genfromtxt(
                    output_files[i][j]
                )[:, -1:].flatten()

        errordata = {}  # creating dictionary for relative errors.
        ferr, axerr = plt.subplots(2, 1)  # figure for relative errors.

        # iterating through, calculating errors and plotting data:
        for i, method in enumerate(methods):
            f, ax = plt.subplots(2, 1)  # figure for numeric solutions.
            for j, N in enumerate(Ns):
                x = np.linspace(0, 1, N+1)  # x-values
                t1 = tdict[method, N][n_t1[j]]  # retrieving t1
                t2 = tdict[method, N][n_t2[j]]  # retrieving t2

                # calculating error
                errordata[method, N] = np.sqrt(np.sum(
                    (data[method, N] -
                     anal_1d(x, tdict[method, N]))**2, axis=1) /
                    np.sqrt(np.sum(anal_1d(x, tdict[method, N])**2, axis=1))
                )

                # plotting results:
                ax[j].plot(x, data[method, N][n_t1[j], :],
                           label=f"Numeric $t_{1} = $ {t1:.3f}")
                ax[j].plot(x, data[method, N][n_t2[j], :],
                           label=f"Numeric $t_{2} = $ {t2:.3f}")

                # plotting analytic solution for comparison:
                x = np.linspace(0, 1, 100)
                ax[j].plot(x, anal_1d(x, t1), '--',
                           label=f"Analytic $t_{1} = $ {t1:.3f}")
                ax[j].plot(x, anal_1d(x, t2), '--',
                           label="Analytic " +
                           f"$t_{2} = $ {t2:.3f}")
                ax[j].set_xlabel(r"$x$")
                ax[j].set_ylabel(r"$u$")
                ax[j].set_title(r"$\Delta x = $" + f"{1/N}" +
                                r"  $\Delta t = $" + f"{dts[j]}")
                ax[j].legend()
                ax[j].grid()

                # plotting reltaive error as function of time steps:
                axerr[j].semilogy(errordata[method, N], '-', label=method)
                axerr[j].set_title(r"$\Delta x = $" + f"{1/N}" +
                                   r"  $\Delta t = $" + f"{dts[j]}")
                axerr[j].set_ylabel(r"$\epsilon(t)$")
                axerr[j].set_xlabel("Time steps $M$")
                axerr[j].legend()
                axerr[j].grid()

            f.suptitle(method)
            f.set_size_inches(10.5/2, 18.5/2)
            f.tight_layout()
            f.savefig(datadir + method + ".pdf")

        ferr.suptitle("Relative RMS error")
        ferr.set_size_inches(10.5/2, 18.5/2)
        ferr.tight_layout()
        ferr.savefig(datadir + "1D-RMS.pdf")

        # print(errordata)
        plt.show()

    if runflag == "2d":
        """Perform data analysis for 2D solver with generic problem."""
        N = 200  # number of grid points minus one along axes.
        M = 50000  # number of time steps to perform.
        dt = 1e-4  # size of time step.
        write_limit = M  # number of steps to skip when writing to file.
        output_filename = datadir + "test2d.dat"
        error_filename = datadir + "error2d.dat"

        if genflag == "y":  # running simulation if needed:
            run_2D(output_filename, N, dt, M, write_limit, error_filename)

        tsteps = int(M/write_limit) + 1  # number of time steps to read in.
        # extracting results and relative error:
        t, data = import_data_2D(output_filename, tsteps, N)
        errordata = np.genfromtxt(error_filename)[:, 0].flatten()
        terr = np.genfromtxt(error_filename)[:, 1].flatten()

        f, (ax1, ax2) = plt.subplots(2, 1)  # creating figure.
        # setting extent of colorbar:
        min, max = np.min(data[0, :, :]), np.max(data[0, :, :])

        # plotting initial state:
        c1 = ax1.imshow(data[0, :, :], vmin=min, vmax=max, cmap='inferno',
                        interpolation='none', origin="lower", aspect='auto',
                        extent=[0, 1, 0, 1])
        ax1.set_title(f"t = {t[0]}")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.grid()

        # plotting final state:
        c2 = ax2.imshow(data[-1, :, :], vmin=min, vmax=max, cmap='inferno',
                        interpolation='none', origin="lower", aspect='auto',
                        extent=[0, 1, 0, 1])
        ax2.set_title(f"t = {t[-1]}")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax2.grid()

        f.colorbar(c1, ax=ax1, label="$u(x,y,t)$")
        f.colorbar(c2, ax=ax2, label="$u(x,y,t)$")
        f.set_size_inches(10.5/2, 18.5/2)
        f.tight_layout()
        f.savefig(datadir + "2D.pdf")

        # plotting relative error as function of time steps:
        ferr, axerr = plt.subplots(1, 1)
        axerr.semilogy(errordata)
        axerr.set_title(r"$\Delta x = $" + f"{1/N}" +
                        r"  $\Delta t = $" + f"{dt}")
        axerr.set_ylabel(r"$\epsilon(t)$")
        axerr.set_xlabel("Time steps $M$")
        axerr.grid()
        ferr.suptitle("Relative RMS error of 2D solver")
        ferr.tight_layout()
        ferr.savefig(datadir + "2Derr.pdf")
        plt.show()

    if runflag in runflags[2:4]:
        """Perform simulation and data analysis of heat problem."""
        # setting simulation parameters:
        N = 300
        Ms = [100000, 10000]
        dt = 1/Ms[1]
        a_x = 2.0            # Gy^1/2
        a_y = 0.8            # Gy^1/2
        write_limits = [Ms[0], Ms[1]]

        # output filenames for data:
        output_filenames = [datadir + "before-enrichment.dat",
                            datadir + "after-enrichment.dat"]

        if genflag == "y":  # running simulation if needed:
            run_heat(output_filenames[0], output_filenames[1], N, dt, Ms[0],
                     write_limits[0], Ms[1], write_limits[1], a_x, a_y)

        f = {}
        ax = {}
        titles = ["Before enrichment", "After enrichment"]
        filenames = ["2D_heat_before.pdf", "2D_heat_after.pdf"]
        for i in range(2):  # reading datasets and plotting results:
            tsteps = int(Ms[i]/write_limits[i]) + 1
            t, data = import_data_2D(output_filenames[i], tsteps, N)

            # calculating boundary conditions for unenriched steady state,
            # this was used to calibrate the simulation:
            p = np.polyfit(np.linspace(0, 1, N+1), data[-1, :, int(N/2)], 2)
            print(p)  # -439.46119612 1634.20540059 81.39770713

            f[i], ax[i] = plt.subplots(2, 1)
            # plotting initial state:
            c1 = ax[i][0].imshow(data[0, :, :], cmap='inferno',
                                 interpolation='none', origin="lower",
                                 aspect='auto', extent=[0, 300, 0, 120])
            ax[i][0].set_title(rf"$t$ = {t[0]} Gy")
            ax[i][0].grid()
            ax[i][0].set_xlabel(r"Width $x$ [km]")
            ax[i][0].set_ylabel(r"Depth $y$ [km]")
            ax[i][0].set_ylim(ax[i][0].get_ylim()[::-1])

            # plotting final state:
            c2 = ax[i][1].imshow(data[-1, :, :], cmap='inferno',
                                 interpolation='none', origin="lower",
                                 aspect='auto', extent=[0, 300, 0, 120])
            ax[i][1].set_title(rf"$t$ = {t[-1]} Gy")
            ax[i][1].grid()
            ax[i][1].set_xlabel(r"Width $x$ [km]")
            ax[i][1].set_ylabel(r"Depth $y$ [km]")
            ax[i][1].set_ylim(ax[i][1].get_ylim()[::-1])

            f[i].colorbar(c1, ax=ax[i][0], label=r"$T$ [$^\circ$C]")
            f[i].colorbar(c2, ax=ax[i][1], label=r"$T$ [$^\circ$C]")

            f[i].suptitle(titles[i])
            f[i].set_size_inches(10.5/2, 18.5/2)
            f[i].tight_layout()
            f[i].savefig(datadir + filenames[i])

        f_diff, ax_diff = plt.subplots()
        c_diff = ax_diff.imshow(data[-1, :, :]-data[0, :, :], cmap='inferno',
                                interpolation='none', origin="lower",
                                aspect='auto', extent=[0, 300, 0, 120])
        ax_diff.set_title("Temperature difference after enrichment")
        ax_diff.grid()
        ax_diff.set_xlabel(r"Width $x$ [km]")
        ax_diff.set_ylabel(r"Depth $y$ [km]")
        ax_diff.set_ylim(ax_diff.get_ylim()[::-1])

        f_diff.colorbar(c_diff, ax=ax_diff, label=r"$T$ [$^\circ$C]")
        f_diff.tight_layout()
        f_diff.savefig(datadir + "2D_heat_temp_diff.pdf")

        plt.show()

if runflag in runflags[6:]:
    """Run benchmarks for each method in 1D and 2D."""
    build_cpp()
    dims = ["1D", "2D"]

    # setting parameters:
    N = 100
    dt = 0.5*0.5/(N)**2
    M = 1000
    write_limit = M

    # boundary conditions for 1D run:
    u_b = 1
    l_b = 0

    # method names and filenames:
    methods = ["ForwardEuler", "BackwardEuler", "CrankNicolson"]
    filename = datadir + "benchmarkrun.dat"

    output = {}
    for method in methods:  # performing N runs for each 1D method:
        times = np.empty(N)
        for i in range(N):
            # running simulation:
            p = Popen(["./main.exe", "1D", f"{N}", f"{dt}", f"{M}",
                       f"{write_limit}", method, filename, f"{u_b}", f"{l_b}"],
                      stdout=PIPE, stderr=PIPE, cwd=src)

            # capturing standard streams and decoding output:
            stdout, stderr = p.communicate()
            outstr = stdout.decode('utf-8')
            # storing time spent in array:
            time = float(outstr.split('=')[1].strip())
            times[i] = time

        # storing mean and standard deviation of time spent:
        output[method] = np.mean(times), np.std(times)

    # printing results of 1D solvers:
    print("Solver benchmarks. N runs.")
    print(f"N = {N}, M = {M}, dt = {dt}, u_b = {u_b}, l_b = {l_b}")
    for m in methods:
        print(f"{m:14}: \u03BC = {output[m][0]:.3e} s," +
              f" \u03C3 = {output[m][1]:.3e} s")

    for i in range(N):  # running N runs of 2D solver:
        p = Popen(["./main.exe", "2D", f"{N}", f"{dt}", f"{M}",
                   f"{write_limit}", filename, "regular"],
                  stdout=PIPE, stderr=PIPE, cwd=src)

        # capturing standard streams and decoding output:
        stdout, stderr = p.communicate()
        outstr = stdout.decode('utf-8')

        # storing time spent in array:
        time = float(outstr.split('=')[1].strip())
        times[i] = time

    # printing mean time spent on 2D solver with standard deviation:
    print(f"{'2D Solver':14}: \u03BC = {np.mean(times):.3e} s," +
          f" \u03C3 = {np.std(times):.3e} s")

    # removing unused datafile generated by simulations:
    run(["rm", "-rf", datadir + "benchmarkrun.dat"])


if runflag in runflags[4:6]:
    """Run unit tests."""
    run(["make", "test"], cwd=src)
    run(["./test_main.exe"], cwd=src)
