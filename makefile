all:
	nvcc cuda_implementation/aco_full.cu -O3 -o ACO_full --std=c++11  -Xcompiler -fopenmp
	nvcc cuda_implementation/aco_reduced.cu -O3 -o ACO_red --std=c++11