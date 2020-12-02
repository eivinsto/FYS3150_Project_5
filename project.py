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

def import_data_2D(file,tsteps,N):
    data = np.genfromtxt(file)
    reformatted_data = np.zeros([tsteps,N,N])
    t = np.zeros(tsteps)
    for k in range(tsteps):
        for ix in range(N):
            reformatted_data[k,:,ix] = data[k,ix*N:(ix+1)*N]
        t[k] = data[k,-1]

    return t, reformatted_data




if __name__=="__main__":
    """
    # 1D sample run

    N = 100
    dt = 1e-5
    M = 100000
    write_limit = 10000
    method = "ForwardEuler"
    output_filename = datadir + "test.dat"
    u_b = 1
    l_b = 0

    build_cpp()
    run(["./main.exe", "1D", f"{N}", f"{dt}", f"{M}", f"{write_limit}", method,
         output_filename, f"{u_b}",f"{l_b}"],cwd=src)

    data = np.genfromtxt(rootdir + "/data/test.dat")

    x = np.linspace(0,1,N+1)

    f, ax = plt.subplots()
    for i in range(len(data[:,0])):
        ax.plot(x,data[i,:],label=f"t = {i*1e-5}")

    ax.legend()
    ax.grid()

    plt.show()
    """

    # 2D sample run

    N = 100
    dt = 1e-4
    M = 10000
    write_limit = 1000
    output_filename = datadir + "test.dat"

    build_cpp()
    run(["./main.exe", "2D", f"{N}", f"{dt}", f"{M}", f"{write_limit}",
         output_filename], cwd=src)


    # Reformat data
    tsteps = int(M/write_limit)
    h = 1/(N+1)
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    X,Y = np.meshgrid(x,y)

    t, data = import_data_2D(output_filename,tsteps,N)

    f, (ax1,ax2) = plt.subplots(1,2)
    c1 = ax1.contourf(X,Y,data[0,:,:])
    ax1.set_title(f"t = {t[0]}")
    ax1.grid()
    c2 = ax2.contourf(X,Y,data[-1,:,:])
    ax2.set_title(f"t = {t[-1]}")
    ax2.grid()
    f.colorbar(c1,ax=ax1)
    f.colorbar(c2,ax=ax2)
    plt.show()
