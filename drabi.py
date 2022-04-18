#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from scipy.linalg import expm, norm


def step(psi, omega, gamma, dt):

    # step the wave function forward

    heff = np.array([[0, omega], [omega, -1j * gamma / 2]])

    pjump = np.abs(psi[1]) ** 2 * gamma * dt
    if random() <= pjump:
        # the atom completes a jump
        psi = np.array([1.0, 0.0])
    else:
        # the atom does not jump
        # evolve under Heff
        psi = expm(-1j * heff * dt) @ psi
        psi = psi / norm(psi)
    
    return psi


def solve_rho(psi0, omega, gamma, steps, trajectories, dt):
    """
    solve a number of trajectories and 
    return the density matrix at all times.
    """

    rho = np.zeros((2, 2, steps + 1), dtype='complex')
    psi = np.zeros(2)

    for i in range(trajectories):
        psi = psi0
        rho[:, :, 0] += np.outer(psi, np.conj(psi)) / trajectories
        for j in range(steps):
            psi = step(psi, omega, gamma, dt)
            rho[:, :, j+1] += np.outer(psi, np.conj(psi)) / trajectories
    
    return rho


def avg_excited_probability(rho):

    # define the projector that will act as the "observable"
    proj = np.array([[0.0, 0.0], [0.0, 1.0]])

    # trace over the projector and density matrix product
    # at every time step
    exc_prob = np.zeros(rho.shape[2])
    for i in range(exc_prob.size):
        exc_prob[i] = np.trace(proj @ rho[:, :, i]).real

    return exc_prob


def output_rho(rho, t, fname):
    """
    ouptut rho as a csv file
    """

    # create a dataframe with one column per density matix entry,
    # and one row per time step. Output as a .csv file.

    df_out = pd.DataFrame(data={
        "t": t,
        "rho00": rho[0, 0, :],
        "rho01": rho[0, 1, :],
        "rho10": rho[1, 0, :],
        "rho11": rho[1, 1, :]
    })


    df_out.to_csv(fname)


def main():

    # read parameters stored in input.csv
    df_in = pd.read_csv("input.csv")

    # set initial wave function and solve
    psi0 = np.array([0.0, 1.0])
    rho = solve_rho(psi0, df_in["omega"][0], df_in["gamma"][0], \
            df_in["steps"][0], df_in["trajectories"][0], df_in["dt"][0])
    t = np.linspace(0.0, df_in["dt"][0] * df_in["steps"][0], num=(df_in["steps"][0] + 1))
    exc_prob = avg_excited_probability(rho)

    # plot results
    plt.plot(t, exc_prob)
    plt.show()
    plt.savefig("plot.jpg")

    # output results to file
    output_rho(rho, t, "ouptut.csv")


if __name__ == "__main__":
    main()
