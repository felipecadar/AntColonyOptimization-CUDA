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

        mean1 = np.nanmean(all_solutions, axis=2)
        mean2 = np.nanmean(mean1, axis=0)
        
        max1 = np.nanmax(all_solutions, axis=2)
        max2 = np.nanmax(max1, axis=0)

        min1 = np.nanmin(all_solutions, axis=2)
        min2 = np.nanmin(min1, axis=0)

        return mean2, max2, min2


if __name__ == "__main__":
    red = glob.glob(sys.argv[1])
    full = glob.glob(sys.argv[2])

    # box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    box_red = []
    box_full = []
    
    mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure

    mean_results = None
    min_results = None
    max_results = None

    for idx, filename in enumerate(red):
        mean2, max2, min2 = read_exp(filename)

        if mean_results is None:
            mean_results = np.zeros([len(red) , mean2.shape[0]])
            min_results = np.zeros([len(red) , mean2.shape[0]])
            max_results = np.zeros([len(red) , mean2.shape[0]])
        
        mean_results[idx] = mean2        
        max_results[idx] = max2
        min_results[idx] = min2

    mean_ax.plot(np.nanmean(max_results, axis=0), label="ACO_red", color="g")


    mean_results = None
    min_results = None
    max_results = None
    
    for idx, filename in enumerate(full):
        mean2, max2, min2 = read_exp(filename)

        if mean_results is None:
            mean_results = np.zeros([len(full) , mean2.shape[0]])
            min_results = np.zeros([len(full) , mean2.shape[0]])
            max_results = np.zeros([len(full) , mean2.shape[0]])

        mean_results[idx] = mean2        
        max_results[idx] = max2
        min_results[idx] = min2

    mean_ax.plot(np.nanmean(max_results, axis=0), label="ACO_full", color="b")

    mean_ax.set_title("Best values per iteration")
    mean_ax.set_xlabel("Iteration")
    mean_ax.set_ylabel("Solution Value")

    mean_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    mean_fig.tight_layout()
    plt.show()
