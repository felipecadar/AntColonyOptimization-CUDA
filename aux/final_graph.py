import sys, os, glob
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

        # mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure

        mean1 = np.nanmean(all_solutions, axis=2)
        mean2 = np.nanmean(mean1, axis=0)
        
        max1 = np.nanmax(all_solutions, axis=2)
        max2 = np.nanmax(max1, axis=0)

        min1 = np.nanmin(all_solutions, axis=2)
        min2 = np.nanmin(min1, axis=0)

        return mean2, max2, min2

        # mean_ax.plot(mean2, label="mean", color="b")
        # mean_ax.plot(max2, label="max", color="g")
        # mean_ax.plot(min2, label="min", color="r")

        # mean_ax.set_title("Best, Mean and Worse values for each Ant")
        # mean_ax.set_xlabel("Iteration")
        # mean_ax.set_ylabel("Solution Value")

        # # mean_fig.legend()
        # mean_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # mean_fig.tight_layout()
        # plt.show()

if __name__ == "__main__":
    files = glob.glob(sys.argv[1])
    
    mean_results = None
    min_results = None
    max_results = None

    max_of_all = [0]
    
    for idx, filename in enumerate(files):
        mean2, max2, min2 = read_exp(filename)

        if mean_results is None:
            mean_results = np.zeros([len(files) , mean2.shape[0]])
            min_results = np.zeros([len(files) , mean2.shape[0]])
            max_results = np.zeros([len(files) , mean2.shape[0]])
        
        if np.max(max2) > np.max(max_of_all):
            max_of_all = max2

        mean_results[idx] = mean2        
        max_results[idx] = max2
        min_results[idx] = min2

    mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure

    mean_ax.plot(np.nanmean(mean_results, axis=0), label="Mean", color="b")
    mean_ax.plot(np.nanmean(max_results, axis=0), label="Best", color="g")
    mean_ax.plot(np.nanmean(min_results, axis=0), label="Worse", color="r")

    mean_ax.plot(max_of_all, color="g", alpha=0.2)

    mean_ax.set_title("Best, Mean and Worse values for each Ant")
    mean_ax.set_xlabel("Iteration")
    mean_ax.set_ylabel("Solution Value")

    mean_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    mean_fig.tight_layout()
    plt.show()

    print(np.max(max_of_all))