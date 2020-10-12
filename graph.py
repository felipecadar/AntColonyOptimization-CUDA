import sys, os
import matplotlib.pyplot as plt
import numpy as np


def read_exp(fname):
    params = {
        "base": "",
        "ants": 0,
        "iter": 0,
        "evap": 0,
        "alpha": 0,
        "beta": 0,
        "n": 0 
    }

    with open(fname, "r") as f:
        lines = f.readlines()
        params["base"] = lines[0].strip().split(" ")[-1].split("/")[-1]
        params["n"] = int(lines[1].strip().split(" ")[-1])
        params["ants"] = int(lines[3].strip().split(" ")[-1])
        params["iter"] = int(lines[4].strip().split(" ")[-1])
        params["evap"] = float(lines[5].strip().split(" ")[-1])
        params["alpha"] = int(lines[6].strip().split(" ")[-1])
        params["beta"] = int(lines[7].strip().split(" ")[-1])

        results = lines[8: 8 + (params["n"] * params["iter"])]

        best_sol = lines[8 + (params["n"] * params["iter"])].strip()
        best_sol = np.fromstring(best_sol, dtype=np.int, sep=' ')
        best_sum = int(lines[8 + (params["n"] * params["iter"]) + 1].strip())

        all_solutions = np.zeros([params["n"], params["iter"], params["ants"]])
        mean_phero = np.zeros([params["n"], params["iter"]])

        err = 0
        for line in results:
            try:
                p,s = line.strip().split(":")
            except:
                err+=1
                continue
    
    
            _, START_NODE, _ , ITER,_ , MEAN_PHERO = p.strip().split(" ")
            
            ITER = int(ITER)
            START_NODE = int(START_NODE)
            MEAN_PHERO = int(MEAN_PHERO)

            mean_phero[START_NODE, ITER] = MEAN_PHERO
            all_solutions[START_NODE, ITER] = np.fromstring(s.strip(), sep=' ')

        mean_phero[mean_phero == 0] = np.nan   
        all_solutions[all_solutions == 0] = np.nan

        return all_solutions, mean_phero, best_sol, best_sum, params

if __name__ == "__main__":
    fname = "/home/cadar/Documents/cuda_aco/exp.txt"
    exp = read_exp(fname)