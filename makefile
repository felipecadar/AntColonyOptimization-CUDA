all:
	nvcc main.cu -O3 -o ACO --std=c++11
	#nvcc main.cu -o ACO
