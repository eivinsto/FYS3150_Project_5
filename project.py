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


def run_2D(filename, N, dt, M, write_limit, errfilename=None):
    build_cpp()
    commandlist = ["./main.exe", "2D", f"{N}", f"{dt}", f"{M}",
                   f"{write_limit}", filename, "regular"]
    if errfilename is not None:
        commandlist.append(errfilename)
    run(commandlist, cwd=src)


def run_heat(filename1, filename2, N, dt, M1, write_limit1, M2, write_limit2, ax, ay):
    build_cpp()
    run(["./main.exe", "2D", f"{N}", f"{dt}", f"{M1}", f"{write_limit1}",
         filename1, "heat", f"{ax}", f"{ay}", f"{M2}", f"{write_limit2}",
         filename2], cwd=src)



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
runflags = ["1d", "2d", "h", "heat", "t", "test", "b", "benchmark"]
if __name__ == "__main__":
    print("Runs: 1D, 2D, [h]eat, [t]est, [b]enchmark")
    while (runflag not in runflags):
        runflag = input("Choose run: ").lower()
        if runflag == "q" or runflag == "quit":
            sys.exit(0)

    if runflag not in runflags[4:]:
        genflag = input("Generate data? y/n: ").lower()

    # 1D sample run
    if runflag == "1d":
        Ns = [10, 100]
        dts = np.asarray([0.5*0.5/(N)**2 for N in Ns])
        T = 0.4
        t1 = 0.04
        t2 = 0.1
        n_t1 = np.asarray(t1/dts, dtype=np.int64)
        n_t2 = np.asarray(t2/dts, dtype=np.int64)
        n_T = np.asarray(T/dts, dtype=np.int64)
        Ms = np.asarray([int(T/dt) for dt in dts])
        write_limit = 1
        methods = ["ForwardEuler", "BackwardEuler", "CrankNicolson"]

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
        tdict = {}
        for i, method in enumerate(methods):
            for j, N in enumerate(Ns):
                data[method, N] = np.genfromtxt(output_files[i][j])[:, :-1]
                tdict[method, N] = np.genfromtxt(
                    output_files[i][j]
                )[:, -1:].flatten()

        errordata = {}
        ferr, axerr = plt.subplots(2, 1)
        for i, method in enumerate(methods):
            f, ax = plt.subplots(2, 1)
            for j, N in enumerate(Ns):
                x = np.linspace(0, 1, N+1)
                t1 = tdict[method, N][n_t1[j]]
                t2 = tdict[method, N][n_t2[j]]

                errordata[method, N] = np.sqrt(np.sum(
                    (data[method, N] -
                     anal_1d(x, tdict[method, N]))**2, axis=1) /
                    np.sqrt(np.sum(anal_1d(x, tdict[method, N])**2, axis=1))
                )

                ax[j].plot(x, data[method, N][n_t1[j], :],
                           label=f"Numeric $t_{1} = $ {t1:.3f}")
                ax[j].plot(x, data[method, N][n_t2[j], :],
                           label=f"Numeric $t_{2} = $ {t2:.3f}")

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

    # 2D sample run
    if runflag == "2d":
        N = 100
        M = 5000
        dt = 1e-4
        write_limit = M
        output_filename = datadir + "test2d.dat"
        error_filename = datadir + "error2d.dat"

        if genflag == "y":
            run_2D(output_filename, N, dt, M, write_limit, error_filename)

        tsteps = int(M/write_limit) + 1
        t, data = import_data_2D(output_filename, tsteps, N)
        errordata = np.genfromtxt(error_filename)[:, 0].flatten()
        terr = np.genfromtxt(error_filename)[:, 1].flatten()

        f, (ax1, ax2) = plt.subplots(2, 1)
        min, max = np.min(data[0, :, :]), np.max(data[0, :, :])
        c1 = ax1.imshow(data[0, :, :], vmin=min, vmax=max, cmap='inferno',
                        interpolation='none', origin="lower", aspect='auto',
                        extent=[0, 1, 0, 1])
        ax1.set_title(f"t = {t[0]}")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.grid()

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
        N = 100
        Ms = [100000,10000]
        dt = 1/Ms[1]
        a_x = 2.0            # Gy^1/2
        a_y = 0.8            # Gy^1/2
        write_limits = [Ms[0],Ms[1]]


        output_filenames = [datadir + "before-enrichment.dat",
                            datadir + "after-enrichment.dat"]


        if genflag == "y":

            run_heat(output_filenames[0], output_filenames[1], N, dt, Ms[0],
                     write_limits[0], Ms[1], write_limits[1], a_x, a_y)


        f = {}
        ax = {}
        titles = ["Before enrichment", "After enrichment"]
        filenames = ["2D_heat_before.pdf", "2D_heat_after.pdf"]
        for i in range(2):
            tsteps = int(Ms[i]/write_limits[i]) + 1
            t, data = import_data_2D(output_filenames[i], tsteps, N)

            f[i], ax[i] = plt.subplots(2, 1)
            c1 = ax[i][0].imshow(data[0, :, :], cmap='inferno', interpolation='none',
                            origin="lower", aspect='auto', extent=[0, 300, 0, 120])
            ax[i][0].set_title(rf"$t$ = {t[0]} Gy")
            ax[i][0].grid()
            ax[i][0].set_xlabel(r"Width $x$ [km]")
            ax[i][0].set_ylabel(r"Depth $y$ [km]")
            ax[i][0].set_ylim(ax[i][0].get_ylim()[::-1])
            c2 = ax[i][1].imshow(data[-1, :, :], cmap='inferno', interpolation='none',
                            origin="lower", aspect='auto', extent=[0, 300, 0, 120])
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
            """
            t2, data2 = import_data_2D(output_filename2, tsteps2, N)

            f2, (ax21, ax22) = plt.subplots(2, 1)
            c21 = ax21.imshow(data2[0, :, :], cmap='inferno', interpolation='none',
                            origin="lower", aspect='auto', extent=[0, 300, 0, 120])
            ax21.set_title(rf"$t$ = {t2[0]} Gy")
            ax21.grid()
            ax21.set_xlabel(r"Width $x$ [km]")
            ax21.set_ylabel(r"Depth $y$ [km]")
            ax21.set_ylim(ax21.get_ylim()[::-1])
            c22 = ax22.imshow(data2[-1, :, :], cmap='inferno', interpolation='none',
                            origin="lower", aspect='auto', extent=[0, 300, 0, 120])
            ax22.set_title(rf"$t$ = {t2[-1]} Gy")
            ax22.grid()
            ax22.set_xlabel(r"Width $x$ [km]")
            ax22.set_ylabel(r"Depth $y$ [km]")
            ax22.set_ylim(ax22.get_ylim()[::-1])

            f2.colorbar(c21, ax=ax21, label=r"$T$ [$^\circ$C]")
            f2.colorbar(c22, ax=ax22, label=r"$T$ [$^\circ$C]")

            f2.suptitle("After enrichment")
            f2.set_size_inches(10.5/2, 18.5/2)
            f2.tight_layout()
            f2.savefig(datadir + "2Dheat_after.pdf")
            """
        plt.show()

if runflag in runflags[6:]:
    print(runflag)

if runflag in runflags[4:6]:
    run(["make", "test"], cwd=src)
    run(["./test_main.exe"], cwd=src)
