import numpy
import sys, os
import argparse

from tqdm import tqdm
import multiprocessing

from utils import *
import ACO



def parseArgs():
    parser = argparse.ArgumentParser("Ant Colony Optimization - Longest Path")
    parser.add_argument("-i", "--input",            type=str, default="bases_grafos/entrada1.txt", required=False, help="Input graph" )
    parser.add_argument("-p", "--pop-size",         type=int, default=10,   required=False, help="Number of ants" )
    parser.add_argument("-a", "--alpha",            type=int, default=1,    required=False, help="Pheromone weight" )
    parser.add_argument("-b", "--beta",             type=int, default=2,    required=False, help="Desirability weight" )
    parser.add_argument("-e", "--evaporation",      type=int, default=0.1,  required=False, help="Pheromone evaporation" )
    parser.add_argument("-m", "--max-iterations",   type=int, default=100,  required=False, help="Max Iterations" )
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parseArgs()

    g = genEnv(args.input)
    initial_nodes = getInitalNodes(g)

    pool = multiprocessing.Pool(processes=os.cpu_count())

    thread_args = []
    for idx, i in enumerate(initial_nodes):
        thread_args.append((idx, i, g, args))

    solutions = pool.map(ACO.runColony, thread_args)
    
    best_sol = []
    best_val = 0

    for ret in solutions:
        sol, val = ret
        if val > best_val:
            best_sol = sol
            best_val = val

    print(chr(27) + "[2J")
    print("Best Solution: ", best_val)
    print(best_sol)