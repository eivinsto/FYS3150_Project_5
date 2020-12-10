from subprocess import run  # , Popen, PIPE
# import multiprocessing as mp
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
         filename, "regular"], cwd=src)


def run_heat(filename, N, dt, M, write_limit, ax, ay, source_type):
    build_cpp()
    run(["./main.exe", "2D", f"{N}", f"{dt}", f"{M}", f"{write_limit}",
         filename, "heat", f"{ax}", f"{ay}", source_type], cwd=src)


def import_data_2D(file, tsteps, N):
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
if __name__ == "__main__":
    print("Runs: 1D, 2D, heat, test")
    while (runflag != "1d" and runflag != "2d" and
           runflag != "heat" and runflag != "test"):
        runflag = input("Choose run: ").lower()
        if runflag == "q" or runflag == "quit":
            sys.exit(0)

    if runflag != "test":
        genflag = input("Generate data? y/n: ").lower()

    # 1D sample run
    if runflag == "1d":
        Ns = [10, 100]
        dts = np.asarray([0.5*0.5/(N)**2 for N in Ns])
        T = 0.1
        t1 = 0.04
        n_t1 = np.asarray(t1/dts, dtype=np.int64)
        n_T = np.asarray(T/dts, dtype=np.int64)
        Ms = np.asarray([int(T/dt - 1) for dt in dts])
        write_limit = 1
        methods = ["ForwardEuler", "BackwardEuler", "CrankNicholson"]

        output_files = []
        for method in methods:
            output_files.append([datadir + method + f"_{N}.dat" for N in Ns])

        u_b = 1
        l_b = 0

        if genflag == "y":
            for i, method in enumerate(methods):
                for j, N in enumerate(Ns):
                    run_1D(output_files[i][j], method, N, dts[j], Ms[j], u_b,
                           l_b, write_limit)

        def anal_1d(x, t):
            """Analytic solution for 1D example"""
            if isinstance(t, np.ndarray):
                arr = np.empty((len(t), len(x)))
                for i in range(len(t)):
                    arr[i, :] = (np.sin(2*np.pi*x)*np.exp(-4*t[i]*np.pi**2) +
                                 x)
                return arr
            else:
                return (np.sin(2*np.pi*x)*np.exp(-4*t*np.pi**2) + x)

        data = {}
        for i, method in enumerate(methods):
            for j, N in enumerate(Ns):
                data[method, N] = np.genfromtxt(output_files[i][j])

        errordata = {}
        for i, method in enumerate(methods):
            ferr, axerr = plt.subplots(1, 2)
            f, ax = plt.subplots(1, 2)
            for j, N in enumerate(Ns):
                x = np.linspace(0, 1, N+1)
                t = np.linspace(0, 1, n_T[j])

                errordata[method, N] = np.sqrt(np.sum(
                    (data[method, N] - anal_1d(x, t))**2, axis=1) /
                    np.sqrt(np.sum(anal_1d(x, t)**2, axis=1))
                )
                idx = np.argmax(errordata[method, N])
                print(idx)

                ax[j].plot(x, data[method, N][idx, :],
                           label=f"Numeric $t_{1} = $ {dts[j]*t[idx]:.3f}")
                ax[j].plot(x, data[method, N][-1, :],
                           label=f"Numeric $t_{2} = $ {dts[j]*n_T[j]:.3f}")

                x = np.linspace(0, 1, 100)
                ax[j].plot(x, anal_1d(x, dts[j]*t[idx]), '--',
                           label=f"Analytic $t_{1} = $ {dts[j]*t[idx]:.3f}")
                ax[j].plot(x, anal_1d(x, dts[j]*n_T[j]), '--',
                           label="Analytic " +
                           f"$t_{1} = $ {dts[j]*n_T[j]:.3f}")
                ax[j].set_title(r"$\Delta x = $ " f"{1/N}")
                ax[j].legend()
                ax[j].grid()

                axerr[j].plot(errordata[method, N], '.')
                axerr[j].set_title(r"$\Delta x = $ " f"{1/N}")
                axerr[j].set_xlabel("Time steps $M$")
                # axerr[j].legend()
                axerr[j].grid()

            f.suptitle(method)
            f.tight_layout()
            f.savefig(datadir + method + ".pdf")

            ferr.suptitle("Mean absolute error " + method)
            ferr.tight_layout()

        # print(errordata)
        plt.show()

    # 2D sample run
    if runflag == "2d":
        N = 100
        M = 10000
        dt = 1/M
        write_limit = M
        output_filename = datadir + "test2d.dat"

        if genflag == "y":
            run_2D(output_filename, N, dt, M, write_limit)

        tsteps = int(M/write_limit) + 1
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
        f.tight_layout()
        f.savefig(datadir + "2Dtest.pdf")
        plt.show()

    if runflag == "heat":
        N = 100
        M = 10000
        dt = 1/M
        a_x = 2.0            # Gy^1/2
        a_y = 0.8            # Gy^1/2
        write_limit = M

        source_type = input("Used enriched source? y/n: ").lower()
        if source_type == "y":
            source_type = "enriched"
        else:
            source_type = "unenriched"

        output_filename = datadir + source_type + "-heat.dat"

        if genflag == "y":

            run_heat(output_filename, N, dt, M, write_limit, a_x, a_y,
                     source_type)

        tsteps = int(M/write_limit) + 1
        t, data = import_data_2D(output_filename, tsteps, N)

        f, (ax1, ax2) = plt.subplots(1, 2)
        c1 = ax1.imshow(data[0, :, :], interpolation='none',
                        origin="lower", aspect='auto', extent=[0, 300, 0, 120])
        ax1.set_title(f"t = {t[0]} Gy")
        ax1.grid()
        ax1.set_xlabel("Width [km]")
        ax1.set_ylabel("Depth [km]")
        ax1.set_ylim(ax1.get_ylim()[::-1])
        c2 = ax2.imshow(data[-1, :, :], interpolation='none',
                        origin="lower", aspect='auto', extent=[0, 300, 0, 120])
        ax2.set_title(f"t = {t[-1]} Gy")
        ax2.grid()
        ax2.set_xlabel("Width [km]")
        ax2.set_ylabel("Depth [km]")
        ax2.set_ylim(ax2.get_ylim()[::-1])
        f.colorbar(c1, ax=ax1)
        f.colorbar(c2, ax=ax2)
        f.suptitle(source_type)
        f.tight_layout()
        f.savefig(datadir + source_type + "2Dheat.pdf")
        plt.show()

if runflag == "test":
    run(["make", "test"], cwd=src)
    run(["./test_main.exe"], cwd=src)
