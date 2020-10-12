import sys, os
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile

import errno    
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def read_exp(fname, zf):
    params = {
        "base": "",
        "ants": 0,
        "iter": 0,
        "evap": 0,
        "alpha": 0,
        "beta": 0,
        "n": 0 
    }

    with zf.open(fname, "r") as f:
        lines = f.readlines()
        lines = [x.decode("utf-8") for x in lines]

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
    zf = ZipFile("results.zip")

    ## Exp params
    # bases = ["bases_grafos/entrada1.txt"]
    bases = ["bases_grafos/entrada1.txt" , "bases_grafos/entrada2.txt"]
    n_iter = [10, 50, 100, 200]
    n_ants = [10, 50, 200, 300]
    evap = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha = [1, 2, 3]
    beta = [1, 2, 3]
    REP = range(30)

    ##################### Exp

    exp_name = "exp_N-ANTS"
    for database in bases:
        box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        mkdir_p("plots/"+dbname)
        box_plot_data = []
        for idx, var in enumerate(n_ants):
            
            mean_final_sol = np.zeros(len(REP))

            mean_sol_p_iter = None
            best_sol_p_iter = None
            worse_sol_p_iter = None
            
            for i in REP:
                exp_id = "results/exp_N-ANTS_database-{}_rep-{:02d}_var-{:03d}.txt".format(dbname, i, var)
                all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id, zf)

                ## If Not created
                if mean_sol_p_iter is None:
                    mean_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    best_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    worse_sol_p_iter = np.zeros([len(REP), params["iter"]])


                mean_final_sol[i] = best_sum
                

                solutions_from_one_node = all_solutions[0]
                mean_sol_p_iter[i] = np.mean(solutions_from_one_node, axis=1)
                best_sol_p_iter[i] = np.max(solutions_from_one_node, axis=1)
                worse_sol_p_iter[i] = np.min(solutions_from_one_node, axis=1)

            mean_ax.plot(np.mean(mean_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            best_ax.plot(np.mean(best_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])

            box_plot_data.append(mean_final_sol)

        box_ax.boxplot(box_plot_data, labels=n_ants)
        box_ax.set_title("Base: " + dbname + " - Distribuição dos resultados finais")
        box_ax.set_xlabel("Número de Formigas")
        box_ax.set_ylabel("Valor da solução")
        box_fig.tight_layout()

        mean_ax.set_title("Base: " +  dbname + " - Média das soluções por iteração")
        mean_ax.set_xlabel("Iteração")
        mean_ax.set_ylabel("Valor da solução")
        mean_ax.legend(title="Formigas", bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()

        best_ax.set_title("Base: " +  dbname + " - Melhor solução por iteração")
        best_ax.set_xlabel("Iteração")
        best_ax.set_ylabel("Valor da solução")
        best_ax.legend(title="Formigas", bbox_to_anchor=(1.05, 1), loc='upper left')
        best_fig.tight_layout()

        worse_ax.set_title("Base: " +  dbname + " - Pior solução por iteração")
        worse_ax.set_xlabel("Iteração")
        worse_ax.set_ylabel("Valor da solução")
        worse_ax.legend(title="Formigas", bbox_to_anchor=(1.05, 1), loc='upper left')
        worse_fig.tight_layout()
        
        box_fig.savefig("plots/{}/{}-box.png".format(dbname, exp_name))
        mean_fig.savefig("plots/{}/{}-mean.png".format(dbname, exp_name))
        best_fig.savefig("plots/{}/{}-best.png".format(dbname, exp_name))
        worse_fig.savefig("plots/{}/{}-worse.png".format(dbname, exp_name))
        plt.close("all")

    ##################### Exp 
    exp_name = "exp_N-ITER"
    for database in bases:
        box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        mkdir_p("plots/"+dbname)
        box_plot_data = []
        for idx, var in enumerate(n_iter):
            
            mean_final_sol = np.zeros(len(REP))

            mean_sol_p_iter = None
            best_sol_p_iter = None
            worse_sol_p_iter = None
            
            for i in REP:
                exp_id = "results/{}_database-{}_rep-{:02d}_var-{:03d}.txt".format(exp_name, dbname, i, var)
                all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id, zf)

                ## If Not created
                if mean_sol_p_iter is None:
                    mean_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    best_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    worse_sol_p_iter = np.zeros([len(REP), params["iter"]])


                mean_final_sol[i] = best_sum
                

                solutions_from_one_node = all_solutions[0]
                mean_sol_p_iter[i] = np.mean(solutions_from_one_node, axis=1)
                best_sol_p_iter[i] = np.max(solutions_from_one_node, axis=1)
                worse_sol_p_iter[i] = np.min(solutions_from_one_node, axis=1)

            mean_ax.plot(np.mean(mean_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            best_ax.plot(np.mean(best_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])

            box_plot_data.append(mean_final_sol)

        box_ax.boxplot(box_plot_data, labels=n_iter)
        box_ax.set_title("Base: " + dbname + " - Distribuição dos resultados finais")
        box_ax.set_xlabel("Número de Iterações")
        box_ax.set_ylabel("Valor da solução")
        box_fig.tight_layout()

        mean_ax.set_title("Base: " +  dbname + " - Média das soluções por iteração")
        mean_ax.set_xlabel("Iteração")
        mean_ax.set_ylabel("Valor da solução")
        mean_ax.legend(title="Iterações", bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()

        best_ax.set_title("Base: " +  dbname + " - Melhor solução por iteração")
        best_ax.set_xlabel("Iteração")
        best_ax.set_ylabel("Valor da solução")
        best_ax.legend(title="Iterações", bbox_to_anchor=(1.05, 1), loc='upper left')
        best_fig.tight_layout()

        worse_ax.set_title("Base: " +  dbname + " - Pior solução por iteração")
        worse_ax.set_xlabel("Iteração")
        worse_ax.set_ylabel("Valor da solução")
        worse_ax.legend(title="Iterações", bbox_to_anchor=(1.05, 1), loc='upper left')
        worse_fig.tight_layout()
        
        box_fig.savefig("plots/{}/{}-box.png".format(dbname, exp_name))
        mean_fig.savefig("plots/{}/{}-mean.png".format(dbname, exp_name))
        best_fig.savefig("plots/{}/{}-best.png".format(dbname, exp_name))
        worse_fig.savefig("plots/{}/{}-worse.png".format(dbname, exp_name))
        plt.close("all")


    ##################### Exp 
    exp_name = "exp_ALPHA"
    for database in bases:
        box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        mkdir_p("plots/"+dbname)
        box_plot_data = []
        for idx, var in enumerate(alpha):
            
            mean_final_sol = np.zeros(len(REP))

            mean_sol_p_iter = None
            best_sol_p_iter = None
            worse_sol_p_iter = None
            
            for i in REP:
                exp_id = "results/{}_database-{}_rep-{:02d}_var-{:03d}.txt".format(exp_name, dbname, i, var)
                all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id, zf)

                ## If Not created
                if mean_sol_p_iter is None:
                    mean_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    best_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    worse_sol_p_iter = np.zeros([len(REP), params["iter"]])


                mean_final_sol[i] = best_sum
                

                solutions_from_one_node = all_solutions[0]
                mean_sol_p_iter[i] = np.mean(solutions_from_one_node, axis=1)
                best_sol_p_iter[i] = np.max(solutions_from_one_node, axis=1)
                worse_sol_p_iter[i] = np.min(solutions_from_one_node, axis=1)

            mean_ax.plot(np.mean(mean_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            best_ax.plot(np.mean(best_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])

            box_plot_data.append(mean_final_sol)

        box_ax.boxplot(box_plot_data, labels=alpha)
        box_ax.set_title("Base: " + dbname + " - Distribuição dos resultados finais")
        box_ax.set_xlabel("Valor do Alpha")
        box_ax.set_ylabel("Valor da solução")
        box_fig.tight_layout()

        mean_ax.set_title("Base: " +  dbname + " - Média das soluções por iteração")
        mean_ax.set_xlabel("Iteração")
        mean_ax.set_ylabel("Valor da solução")
        mean_ax.legend(title="Alpha", bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()

        best_ax.set_title("Base: " +  dbname + " - Melhor solução por iteração")
        best_ax.set_xlabel("Iteração")
        best_ax.set_ylabel("Valor da solução")
        best_ax.legend(title="Alpha", bbox_to_anchor=(1.05, 1), loc='upper left')
        best_fig.tight_layout()

        worse_ax.set_title("Base: " +  dbname + " - Pior solução por iteração")
        worse_ax.set_xlabel("Iteração")
        worse_ax.set_ylabel("Valor da solução")
        worse_ax.legend(title="Alpha", bbox_to_anchor=(1.05, 1), loc='upper left')
        worse_fig.tight_layout()
        
        box_fig.savefig("plots/{}/{}-box.png".format(dbname, exp_name))
        mean_fig.savefig("plots/{}/{}-mean.png".format(dbname, exp_name))
        best_fig.savefig("plots/{}/{}-best.png".format(dbname, exp_name))
        worse_fig.savefig("plots/{}/{}-worse.png".format(dbname, exp_name))
        plt.close("all")



    ##################### Exp 
    exp_name = "exp_BETA"
    for database in bases:
        box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        mkdir_p("plots/"+dbname)
        box_plot_data = []
        for idx, var in enumerate(beta):
            
            mean_final_sol = np.zeros(len(REP))

            mean_sol_p_iter = None
            best_sol_p_iter = None
            worse_sol_p_iter = None
            
            for i in REP:
                exp_id = "results/{}_database-{}_rep-{:02d}_var-{:03d}.txt".format(exp_name, dbname, i, var)
                all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id, zf)

                ## If Not created
                if mean_sol_p_iter is None:
                    mean_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    best_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    worse_sol_p_iter = np.zeros([len(REP), params["iter"]])


                mean_final_sol[i] = best_sum
                

                solutions_from_one_node = all_solutions[0]
                mean_sol_p_iter[i] = np.mean(solutions_from_one_node, axis=1)
                best_sol_p_iter[i] = np.max(solutions_from_one_node, axis=1)
                worse_sol_p_iter[i] = np.min(solutions_from_one_node, axis=1)

            mean_ax.plot(np.mean(mean_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            best_ax.plot(np.mean(best_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])

            box_plot_data.append(mean_final_sol)

        box_ax.boxplot(box_plot_data, labels=beta)
        box_ax.set_title("Base: " + dbname + " - Distribuição dos resultados finais")
        box_ax.set_xlabel("Valor do Beta")
        box_ax.set_ylabel("Valor da solução")
        box_fig.tight_layout()

        mean_ax.set_title("Base: " +  dbname + " - Média das soluções por iteração")
        mean_ax.set_xlabel("Iteração")
        mean_ax.set_ylabel("Valor da solução")
        mean_ax.legend(title="Beta", bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()

        best_ax.set_title("Base: " +  dbname + " - Melhor solução por iteração")
        best_ax.set_xlabel("Iteração")
        best_ax.set_ylabel("Valor da solução")
        best_ax.legend(title="Beta", bbox_to_anchor=(1.05, 1), loc='upper left')
        best_fig.tight_layout()

        worse_ax.set_title("Base: " +  dbname + " - Pior solução por iteração")
        worse_ax.set_xlabel("Iteração")
        worse_ax.set_ylabel("Valor da solução")
        worse_ax.legend(title="Beta", bbox_to_anchor=(1.05, 1), loc='upper left')
        worse_fig.tight_layout()
        
        box_fig.savefig("plots/{}/{}-box.png".format(dbname, exp_name))
        mean_fig.savefig("plots/{}/{}-mean.png".format(dbname, exp_name))
        best_fig.savefig("plots/{}/{}-best.png".format(dbname, exp_name))
        worse_fig.savefig("plots/{}/{}-worse.png".format(dbname, exp_name))
        plt.close("all")



    ##################### Exp 
    exp_name = "exp_EVAP"
    for database in bases:
        box_fig, box_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        mean_fig, mean_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        best_fig, best_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        worse_fig, worse_ax = plt.subplots(figsize=[8,4]) ## Create Figure
        phero_fig, phero_ax = plt.subplots(figsize=[8,4]) ## Create Figure
    
        dbname = database.split("/")[-1].split(".")[0]
        mkdir_p("plots/"+dbname)
        box_plot_data = []
        for idx, var in enumerate(evap):
            
            mean_final_sol = np.zeros(len(REP))
            all_mean_phero = np.zeros(len(REP))

            mean_sol_p_iter = None
            best_sol_p_iter = None
            worse_sol_p_iter = None
            mean_phero_p_iter = None
            
            for i in REP:
                exp_id = "results/{}_database-{}_rep-{:02d}_var-{:.1f}.txt".format(exp_name, dbname, i, var)
                all_solutions, mean_phero, best_sol, best_sum, params = read_exp(exp_id, zf)

                ## If Not created
                if mean_sol_p_iter is None:
                    mean_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    best_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    worse_sol_p_iter = np.zeros([len(REP), params["iter"]])
                    mean_phero_p_iter = np.zeros([len(REP), params["iter"]])

                mean_final_sol[i] = best_sum
                mean_phero_p_iter[i] = mean_phero[0] # Only solutions started from node 0

                solutions_from_one_node = all_solutions[0]
                mean_sol_p_iter[i] = np.mean(solutions_from_one_node, axis=1)
                best_sol_p_iter[i] = np.max(solutions_from_one_node, axis=1)
                worse_sol_p_iter[i] = np.min(solutions_from_one_node, axis=1)

            mean_ax.plot(np.mean(mean_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            best_ax.plot(np.mean(best_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            worse_ax.plot(np.mean(worse_sol_p_iter, axis=0), label=var, color=colors[idx % len(colors)])
            phero_ax.plot(np.mean(mean_phero_p_iter, axis=0), label=var, color=colors[idx % len(colors)])

            box_plot_data.append(mean_final_sol)

        box_ax.boxplot(box_plot_data, labels=evap)
        box_ax.set_title("Base: " + dbname + " - Distribuição dos resultados finais")
        box_ax.set_xlabel("Valor do Evaporação")
        box_ax.set_ylabel("Valor da solução")
        box_fig.tight_layout()

        mean_ax.set_title("Base: " +  dbname + " - Média das soluções por iteração")
        mean_ax.set_xlabel("Iteração")
        mean_ax.set_ylabel("Valor da solução")
        mean_ax.legend(title="Evaporação", bbox_to_anchor=(1.05, 1), loc='upper left')
        mean_fig.tight_layout()

        best_ax.set_title("Base: " +  dbname + " - Melhor solução por iteração")
        best_ax.set_xlabel("Iteração")
        best_ax.set_ylabel("Valor da solução")
        best_ax.legend(title="Evaporação", bbox_to_anchor=(1.05, 1), loc='upper left')
        best_fig.tight_layout()

        worse_ax.set_title("Base: " +  dbname + " - Pior solução por iteração")
        worse_ax.set_xlabel("Iteração")
        worse_ax.set_ylabel("Valor da solução")
        worse_ax.legend(title="Evaporação", bbox_to_anchor=(1.05, 1), loc='upper left')
        worse_fig.tight_layout()

        phero_ax.set_title("Base: " +  dbname + " - Média de feromônio por iteração")
        phero_ax.set_xlabel("Iteração")
        phero_ax.set_ylabel("Média de feromônio")
        phero_ax.legend(title="Evaporação", bbox_to_anchor=(1.05, 1), loc='upper left')
        phero_fig.tight_layout()
        
        box_fig.savefig("plots/{}/{}-box.png".format(dbname, exp_name))
        mean_fig.savefig("plots/{}/{}-mean.png".format(dbname, exp_name))
        best_fig.savefig("plots/{}/{}-best.png".format(dbname, exp_name))
        worse_fig.savefig("plots/{}/{}-worse.png".format(dbname, exp_name))
        phero_fig.savefig("plots/{}/{}-phero.png".format(dbname, exp_name))
        plt.close("all")