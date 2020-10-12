all:
	nvcc cuda_implementation/main.cu -O3 -o ACO --std=c++11  -Xcompiler -fopenmp