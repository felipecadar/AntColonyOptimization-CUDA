import os

bases = ["bases_grafos/entrada1.txt" , "bases_grafos/entrada2.txt" , "bases_grafos/entrada3.txt"]
n_iter = [10, 50, 100, 200]
n_ants = [10, 50, 200, 300]
evap = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha = [1, 2, 3]
beta = [1, 2, 3]
REP = range(30)

## Default Args
N_ANTS= 100
N_ITER= 100
EVAP= 0.2
ALPHA= 1
BETA= 1


##################### Exp 
exp_file = open("exp_N-ANTS.sh", "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in n_ants:
        for i in REP:
            exp_id = "exp_N-ANTS_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, var, EVAP, ALPHA, BETA, exp_id)
            exp_file.write(cmd)

##################### Exp 
exp_file = open("exp_N-ITER.sh", "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in n_iter:
        for i in REP:
            exp_id = "exp_N-ITER_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, var, N_ANTS, EVAP, ALPHA, BETA, exp_id)
            exp_file.write(cmd)

##################### Exp 
exp_file = open("exp_ALPHA.sh", "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in alpha:
        for i in REP:
            exp_id = "exp_ALPHA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, EVAP, var, BETA, exp_id)
            exp_file.write(cmd)

##################### Exp 
exp_file = open("exp_BETA.sh", "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in beta:
        for i in REP:
            exp_id = "exp_BETA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, EVAP, ALPHA, var, exp_id)
            exp_file.write(cmd)

##################### Exp 
exp_file = open("exp_EVAP.sh", "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in evap:
        for i in REP:
            exp_id = "exp_EVAP_database-{}_rep-{:02d}_var-{:.1f}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, var, ALPHA, BETA, exp_id)
            exp_file.write(cmd)