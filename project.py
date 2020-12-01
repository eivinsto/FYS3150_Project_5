from subprocess import run, Popen, PIPE
import multiprocessing as mp
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"
data = rootdir + "/data/"


def build_cpp():
    """Function building cpp program."""
    run(["make", "all"], cwd=src)



if __name__=="__main__":
    N = 100
    dt = 1e-5
    M = 100000
    write_limit = 10000
    method = "ForwardEuler"
    output_filename = data + "test.dat"
    u_b = 1
    l_b = 0

    build_cpp()
    run(["./main.exe", f"{N}", f"{dt}", f"{M}", f"{write_limit}", method,
         output_filename, f"{u_b}",f"{l_b}"],cwd=src)

    data = np.genfromtxt(rootdir + "/data/test.dat")

    x = np.linspace(0,1,N+1)

    f, ax = plt.subplots()
    for i in range(len(data[:,0])):
        ax.plot(x,data[i,:],label=f"t = {i*1e-5}")

    ax.legend()
    ax.grid()

    plt.show()
