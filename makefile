all:
	nvcc main.cu -o ACO
	# nvcc main.cu -O3 -o ACO