import sys, os
import numpy as np
import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

        results = lines[8:]

        doing = np.zeros(params["n"])

        all_solutions = np.zeros([params["n"], params["iter"], params["ants"]])
        mean_phero = np.zeros([params["n"], params["iter"]])

        err = 0
        for line in results:
            if "START_NODE" in line:
                try:
                    p,s = line.strip().split(":")
                except:
                    err+=1
                    continue
        
                    
                _, START_NODE, _ , ITER,_ , MEAN_PHERO = p.strip().split(" ")
            
                ITER = int(ITER)    
                START_NODE = int(START_NODE)
                MEAN_PHERO = float(MEAN_PHERO)

                doing[int(START_NODE)] = 1

                mean_phero[START_NODE, ITER] = MEAN_PHERO
                all_solutions[START_NODE, ITER] = np.fromstring(s.strip(), sep=' ')

        mean_phero[mean_phero == 0] = np.nan   
        all_solutions[all_solutions == 0] = np.nan

        for idx, r in enumerate(doing):
            # if idx%40 == 0: print("")
            if r :
                print(bcolors.OKGREEN, end="")
            else:
                print(bcolors.FAIL, end="")
            print("*", end="")
            print(bcolors.ENDC, end="")


        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        # best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        # worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure

        # for n in np.where(doing == 1)[0]:
        #     if n == 0:
        #         mean_ax.plot(np.mean(all_solutions[n], axis=1), color="b", label="mean")
        #         mean_ax.plot(np.max(all_solutions[n], axis=1), color="g", label="best")
        #         mean_ax.plot(np.min(all_solutions[n], axis=1), color="r", label="worse")
        #     else:
        #         mean_ax.plot(np.mean(all_solutions[n], axis=1), color="b", alpha=0.2)
        #         mean_ax.plot(np.max(all_solutions[n], axis=1), color="g", alpha=0.2)
        #         mean_ax.plot(np.min(all_solutions[n], axis=1), color="r", alpha=0.2)

        mean1 = np.nanmean(all_solutions, axis=2)
        mean2 = np.nanmean(mean1, axis=0)
        
        max1 = np.nanmax(all_solutions, axis=2)
        max2 = np.nanmax(max1, axis=0)

        min1 = np.nanmin(all_solutions, axis=2)
        min2 = np.nanmin(min1, axis=0)

        mean_ax.plot(mean2, label="mean", color="b")
        mean_ax.plot(max2, label="max", color="g")
        mean_ax.plot(min2, label="min", color="r")

        mean_ax.set_title("Best, Mean and Worse values for each Ant")
        mean_ax.set_xlabel("Iteration")
        mean_ax.set_ylabel("Solution Value")

        # mean_fig.legend()
        mean_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()
        plt.show()

filename = sys.argv[1]
read_exp(filename)