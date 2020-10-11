all:
	nvcc main.cu -O3 -o ACO_parallel --std=c++11  -Xcompiler -fopenmp
	nvcc old_main.cu -O3 -o ACO --std=c++11