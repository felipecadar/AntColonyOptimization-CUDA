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

def checkfile(exp_id):
    exp_id_fn = "results/" + exp_id + ".txt"
    if os.path.isfile(exp_id_fn):
        line = subprocess.check_output(['tail', '-1', exp_id_fn])
        line = line.decode("utf-8")
        if "START" in line:
            return False
        if len(line) < 20:
            return False

        return True
    else:
        return False


bases = ["bases_grafos/entrada1.txt" , "bases_grafos/entrada2.txt"]
n_iter = [10, 50, 100, 200]
n_ants = [10, 50, 200, 300]
evap = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha = [0, 1, 2, 3]
beta = [0, 1, 2, 3]
REP = range(30)

## Default Args
N_ANTS= 100
N_ITER= 100
EVAP= 0.2
ALPHA= 1
BETA= 1


VERIFY = True
if len(sys.argv) > 1:
    if sys.argv[1] == "v":
        VERIFY = False

ONLYCHECK = False
if len(sys.argv) > 1:
    if sys.argv[1] == "c":
        ONLYCHECK = True

##################### Exp 
c = 0
t = 0
exp_name = "exp_N-ANTS"
if not ONLYCHECK: exp_file = open("{}.sh".format(exp_name), "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in n_ants:
        for i in REP:
            t += 1
            exp_id = "exp_N-ANTS_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, var, EVAP, ALPHA, BETA, exp_id)
            runned = checkfile(exp_id)
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
if not ONLYCHECK: exp_file.close()
if (c == t): print(bcolors.OKGREEN, end="")
print("[{:10}] - {} of {}".format(exp_name, c, t))
if (c == t): print(bcolors.ENDC, end="")
##################### Exp 
c = 0
t = 0
exp_name = "exp_N-ITER"
if not ONLYCHECK: exp_file = open("{}.sh".format(exp_name), "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in n_iter:
        for i in REP:
            t += 1
            exp_id = "exp_N-ITER_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, var, N_ANTS, EVAP, ALPHA, BETA, exp_id)
            runned = checkfile(exp_id)
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
if not ONLYCHECK: exp_file.close()
if (c == t): print(bcolors.OKGREEN, end="")
print("[{:10}] - {} of {}".format(exp_name, c, t))
if (c == t): print(bcolors.ENDC, end="")
##################### Exp 
c = 0
t = 0
exp_name = "exp_ALPHA"
if not ONLYCHECK: exp_file = open("{}.sh".format(exp_name), "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in alpha:
        for i in REP:
            t += 1
            exp_id = "exp_ALPHA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, EVAP, var, BETA, exp_id)
            runned = checkfile(exp_id)
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
if not ONLYCHECK: exp_file.close()
if (c == t): print(bcolors.OKGREEN, end="")
print("[{:10}] - {} of {}".format(exp_name, c, t))
if (c == t): print(bcolors.ENDC, end="")
##################### Exp 
c = 0
t = 0
exp_name = "exp_BETA"
if not ONLYCHECK: exp_file = open("{}.sh".format(exp_name), "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in beta:
        for i in REP:
            t += 1
            exp_id = "exp_BETA_database-{}_rep-{:02d}_var-{:03d}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, EVAP, ALPHA, var, exp_id)
            runned = checkfile(exp_id)
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
if not ONLYCHECK: exp_file.close()
if (c == t): print(bcolors.OKGREEN, end="")
print("[{:10}] - {} of {}".format(exp_name, c, t))
if (c == t): print(bcolors.ENDC, end="")
##################### Exp 
c = 0
t = 0
exp_name = "exp_EVAP"
if not ONLYCHECK: exp_file = open("{}.sh".format(exp_name), "w")
for database in bases:
    dbname = database.split("/")[-1].split(".")[0]
    for var in evap:
        for i in REP:
            t += 1
            exp_id = "exp_EVAP_database-{}_rep-{:02d}_var-{:.1f}".format(dbname, i, var)
            cmd = "./ACO {} {} {} {} {} {} {}\n".format(database, N_ITER, N_ANTS, var, ALPHA, BETA, exp_id)
            runned = checkfile(exp_id)
            if runned: c += 1
            if not ONLYCHECK:
                if VERIFY:
                    if not runned:
                        exp_file.write(cmd)
                else:
                    exp_file.write(cmd)
if not ONLYCHECK: exp_file.close()
if (c == t): print(bcolors.OKGREEN, end="")
print("[{:10}] - {} of {}".format(exp_name, c, t))
if (c == t): print(bcolors.ENDC, end="")
