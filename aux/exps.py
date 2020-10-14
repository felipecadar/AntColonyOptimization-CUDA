import os, sys
import subprocess

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



import errno    
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def checkfile(exp_id):
    exp_id_fn = "results/" + exp_id + ".txt"
    if os.path.isfile(exp_id_fn):
        line = subprocess.check_output(['tail', '-1', exp_id_fn])
        line = line.decode("utf-8")
        if not line.strip().isnumeric():
            return False, True

        return True, False
    else:
        return False, False



variables = {
    "bases_grafos/entrada1.txt" : {
        "n_iter" : [10, 50, 100, 200],
        "n_ants" : [10, 50, 200, 300],
        "evap" : [0.1, 0.3, 0.5, 0.7, 0.9],
        "alpha" : [1, 2, 3],
        "beta" : [1, 2, 3],
    },
    "bases_grafos/entrada2.txt" : {
        "n_iter" : [10, 100, 200],
        "n_ants" : [100, 500, 1000, 2000],
        "evap" : [0.1, 0.3, 0.5, 0.7, 0.9],
        "alpha" : [1, 2, 3],
        "beta" : [1, 2, 3],
    },
    "bases_grafos/entrada3.txt" : {
        "n_iter" : [10, 50, 100, 200],
        "n_ants" : [100, 500, 1000, 2000],
        "evap" : [0.1, 0.3, 0.5, 0.7, 0.9],
        "alpha" : [1, 2, 3],
        "beta" : [1, 2, 3],
    },
}

bases = list(variables.keys())
REP = range(30)

## Default Args
N_ANTS= 100
N_ITER= 100
EVAP= 0.2
ALPHA= 1
BETA= 1
EXEC = "ACO_red"


VERIFY = True
if len(sys.argv) > 1:
    if sys.argv[1] == "v":
        VERIFY = False

ONLYCHECK = False
if len(sys.argv) > 1:
    if sys.argv[1] == "c":
        ONLYCHECK = True

if not ONLYCHECK: mkdir_p("run_exps")

##################### Exp 
exp_name = "exp_N-ANTS"
for database in bases:
    c = 0
    t = 0
    running_now = False
    dbname = database.split("/")[-1].split(".")[0]
    if not ONLYCHECK: exp_file = open("run_exps/{}-{}.sh".format(exp_name, dbname), "w")
    for var in variables[database]["n_ants"]:
        for i in REP:
            t += 1
            exp_id = "exp_N-ANTS_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./{} {} {} {} {} {} {} {}\n".format(EXEC, database, N_ITER, var, EVAP, ALPHA, BETA, exp_id)
            runned, r_now = checkfile(exp_id)
            if r_now: running_now = True
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
    if not ONLYCHECK: exp_file.close()

    if (c == t): print(bcolors.OKGREEN, end="") 
    elif running_now: print(bcolors.WARNING, end="") 
    print("{} - [{:10}] - {} of {}".format(database, exp_name, c, t))
    print(bcolors.ENDC, end="")
##################### Exp 
exp_name = "exp_N-ITER"
for database in bases:
    c = 0
    t = 0
    running_now = False
    dbname = database.split("/")[-1].split(".")[0]
    if not ONLYCHECK: exp_file = open("run_exps/{}-{}.sh".format(exp_name, dbname), "w")
    for var in variables[database]["n_iter"]:
        for i in REP:
            t += 1
            exp_id = "exp_N-ITER_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./{} {} {} {} {} {} {} {}\n".format(EXEC, database, var, N_ANTS, EVAP, ALPHA, BETA, exp_id)
            runned, r_now = checkfile(exp_id)
            if r_now: running_now = True
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
    if not ONLYCHECK: exp_file.close()

    if (c == t): print(bcolors.OKGREEN, end="") 
    elif running_now: print(bcolors.WARNING, end="") 
    print("{} - [{:10}] - {} of {}".format(database, exp_name, c, t))
    print(bcolors.ENDC, end="")

##################### Exp 
exp_name = "exp_ALPHA"
for database in bases:
    c = 0
    t = 0

    running_now = False
    dbname = database.split("/")[-1].split(".")[0]
    if not ONLYCHECK: exp_file = open("run_exps/{}-{}.sh".format(exp_name, dbname), "w")
    for var in variables[database]["alpha"]:
        for i in REP:
            t += 1
            exp_id = "exp_ALPHA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./{} {} {} {} {} {} {} {}\n".format(EXEC, database, N_ITER, N_ANTS, EVAP, var, BETA, exp_id)
            runned, r_now = checkfile(exp_id)
            if r_now: running_now = True
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
    if not ONLYCHECK: exp_file.close()

    if (c == t): print(bcolors.OKGREEN, end="") 
    elif running_now: print(bcolors.WARNING, end="") 
    print("{} - [{:10}] - {} of {}".format(database, exp_name, c, t))
    print(bcolors.ENDC, end="")

##################### Exp 
exp_name = "exp_BETA"
for database in bases:
    c = 0
    t = 0
    running_now = False
    dbname = database.split("/")[-1].split(".")[0]
    if not ONLYCHECK: exp_file = open("run_exps/{}-{}.sh".format(exp_name, dbname), "w")
    for var in variables[database]["beta"]:
        for i in REP:
            t += 1
            exp_id = "exp_BETA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./{} {} {} {} {} {} {} {}\n".format(EXEC, database, N_ITER, N_ANTS, EVAP, ALPHA, var, exp_id)
            runned, r_now = checkfile(exp_id)
            if r_now: running_now = True
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
    if not ONLYCHECK: exp_file.close()

    if (c == t): print(bcolors.OKGREEN, end="") 
    elif running_now: print(bcolors.WARNING, end="") 
    print("{} - [{:10}] - {} of {}".format(database, exp_name, c, t))
    print(bcolors.ENDC, end="")
##################### Exp 
exp_name = "exp_EVAP"
for database in bases:
    c = 0
    t = 0
    running_now = False
    dbname = database.split("/")[-1].split(".")[0]
    if not ONLYCHECK: exp_file = open("run_exps/{}-{}.sh".format(exp_name, dbname), "w")
    for var in variables[database]["evap"]:
        for i in REP:
            t += 1
            exp_id = "exp_EVAP_database-{}_rep-{:02d}_var-{:.1f}".format(dbname, i, var)
            cmd = "./{} {} {} {} {} {} {} {}\n".format(EXEC, database, N_ITER, N_ANTS, var, ALPHA, BETA, exp_id)
            runned, r_now = checkfile(exp_id)
            if r_now: running_now = True
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
    if not ONLYCHECK: exp_file.close()

    if (c == t): print(bcolors.OKGREEN, end="") 
    elif running_now: print(bcolors.WARNING, end="") 
    print("{} - [{:10}] - {} of {}".format(database, exp_name, c, t))
    print(bcolors.ENDC, end="")
