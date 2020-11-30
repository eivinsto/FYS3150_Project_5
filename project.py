from subprocess import run, Popen, PIPE
import multiprocessing as mp
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Retrieving working directories
rootdir = os.getcwd()
src = rootdir + "/src/"


def build_cpp():
    """Function building cpp program."""
    run(["make", "all"], cwd=src)



if __name__=="__main__":
    build_cpp()
    run(["./main.exe"],cwd=src)

    data = np.genfromtxt(rootdir + "/data/test.dat")

    x = np.linspace(0,1,101)

    f, ax = plt.subplots()
    for i in range(len(data[:,0])):
        ax.plot(x,data[i,:],label=f"t = {i*1e-5}")

    ax.legend()
    ax.grid()

    plt.show()
