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
            MEAN_PHERO = float(MEAN_PHERO)

            mean_phero[START_NODE, ITER] = MEAN_PHERO
            all_solutions[START_NODE, ITER] = np.fromstring(s.strip(), sep=' ')

        mean_phero[mean_phero == 0] = np.nan   
        all_solutions[all_solutions == 0] = np.nan

        return all_solutions, mean_phero, best_sol, best_sum, params

if __name__ == "__main__":

    colors = ["#0F54FF","#e41a1c","#984ea3","#377eb8","#ff7f00","#f781bf","#0FD0FF","#a65628","#dede00"]
    
    ## Exp params
    bases = ["bases_grafos/entrada1.txt" , "bases_grafos/entrada2.txt"]
    n_iter = [10, 50, 100, 200]
    n_ants = [10, 50, 200, 300]
    evap = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha = [0, 1, 2, 3]
    beta = [0, 1, 2, 3]
    REP = range(30)

    ##################### Exp

    exp_name = "exp_N-ANTS"
    for database in bases:
        plt.figure() ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        b_plot = []
        for idx, var in enumerate(n_ants):
            
            mean_best_sol = np.zeros(len(REP))

            for i in REP:
                exp_id = "TEST_PLOT/exp_N-ANTS_database-{}_rep-{:02d}_var-{:03d}.txt".format(dbname, i, var)
                try:
                    all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id)
                except:
                    print(exp_id)

                mean_best_sol[i] = best_sum

                # plt.scatter(x = idx ,y=best_sum, color=colors[idx % len(colors)], alpha=0.3)
            
            # plt.scatter(x = idx ,y=np.mean(mean_best_sol), color=colors[idx % len(colors)], label=var)
            b_plot.append(mean_best_sol)

        plt.boxplot(b_plot, labels=n_ants)

        plt.title(exp_name + " - " + dbname)
        plt.xlabel("Número de Formigas")
        plt.ylabel("Valor da solução")
        plt.show()

    exit()
    ##################### Exp 
    exp_name = "exp_N-ITER"
    for database in bases:
        dbname = database.split("/")[-1].split(".")[0]
        for var in n_iter:
            for i in REP:
                exp_id = "results/exp_N-ITER_database-{}_rep-{:02d}_var-{:03d}.txt".format(dbname, i, var)

    ##################### Exp 
    exp_name = "exp_ALPHA"
    for database in bases:
        dbname = database.split("/")[-1].split(".")[0]
        for var in alpha:
            for i in REP:
                exp_id = "results/exp_ALPHA_database-{}_rep-{:02d}_var-{:03d}.txt".format(dbname, i, var)

    ##################### Exp 
    exp_name = "exp_BETA"
    for database in bases:
        dbname = database.split("/")[-1].split(".")[0]
        for var in beta:
            for i in REP:
                exp_id = "results/exp_BETA_database-{}_rep-{:02d}_var-{:03d}.txt".format(dbname, i, var)

    ##################### Exp 
    exp_name = "exp_EVAP"
    for database in bases:
        dbname = database.split("/")[-1].split(".")[0]
        for var in evap:
            for i in REP:
                exp_id = "results/exp_EVAP_database-{}_rep-{:02d}_var-{:.1f}.txt".format(dbname, i, var)
