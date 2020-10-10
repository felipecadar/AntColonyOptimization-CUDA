#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unistd.h>
#include <math.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define THREADS_P_BLOCK 128

#define alpha 1
#define beta 2
#define MAX_WEIGHT 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* this GPU kernel function is used to initialize the random states */
__global__ void rand_init(unsigned int seed, curandState_t* states) {
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
  }


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

__global__ void reset_float(float *v, long int N, float val){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        v[i] = val;
}

__global__ void reset_int(int *v, int N, int val){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        v[i] = val;
    }
}

__global__ void evaporate(float *t, float p, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        t[i] = t[i] * (1-p);
        if(t[i] < 1) t[i] = 1;
    }
}

__global__ void printmat(float *t, int N){
    printf("\n");
    for (int i = 0; i < N; i += 1){
        for (int j = 0; j < N; j += 1){
            printf("%.2f ", t[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void update_trail(float *t, int N, int N_ANTS, int* d_sol, int* sum){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N_ANTS; i += stride){
        int ant_id = i;
        int *sol = &d_sol[ant_id*N];
        
        // for(int idx_sol = 0; idx_sol < N; idx_sol++){
        //     printf("%i ", sol[idx_sol]);
        // }

        // printf("\n");
        for(int idx_sol = 1; idx_sol < N; idx_sol++){
            if(sol[idx_sol] == -1){
                break;
            }
            int from = sol[idx_sol-1];
            int to = sol[idx_sol];
            t[from*N + to] += sum[ant_id];
            // printf("Update trail %i %i %i\n", from, to, sum[ant_id]);
        }
    }
}

__global__ void ant(curandState_t* states, float *t, int *g, int N, int N_ANTS, int init, int *d_sol, int *d_sum, int *d_visited){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int ant_id = index; ant_id < N_ANTS; ant_id += stride){

        // int ant_id = i;
        int *visited = &(d_visited[ant_id*N]);
        int *sol = &(d_sol[ant_id*N]);
        
        
        
        int now_node = init;
        int end = 0;
        int sol_idx = 0;
        
        float *probs = new float[N];
        int *choices = new int[100];
        
        sol[sol_idx] = now_node;
        visited[now_node] = 1;
        
        
        
        while(end == 0){
            sol_idx++;
            
            //Calc Probs
            end = 1;
            float norm = 0;
            for(int neigh = 0; neigh < N; neigh ++){
                probs[neigh] = 0;
                if(g[now_node*N + neigh] > 0 && visited[neigh] == 0){
                    float Tij = t[now_node*N + neigh];
                    float Nij = (float)g[now_node*N + neigh] / (float)MAX_WEIGHT;
                    
                    float res = pow(Tij, alpha) * pow(Nij, beta);
                    
                    probs[neigh] = res;
                    norm += res;
                    
                    end = 0;
                }
            }

            // printf("Norm: %f\n", norm);
            
            //Norm probs to sum 1
            if(end) break;
            
            int idx = 0;
            for(int neigh = 0; neigh < N; neigh ++){ 
                probs[neigh] = (probs[neigh] / norm) * 100;
                
                for(int h = 0; h < (int)probs[neigh]; h++ ){
                    choices[h + idx] = neigh;
                }
                idx += (int)probs[neigh];
            }
            
            // Select random neigh
            int c = curand(&states[blockIdx.x]) % 100;
            int target = choices[c];
            
            d_sum[ant_id] += g[now_node*N + target];
            
            sol[sol_idx] = target;
            visited[target] = 1;
            
            now_node = target;
            

        }

        // if(d_sum[ant_id] >= 177){
        //     printf("------------------------------------------------\n[");
        //     for(int idx_sol = 0; idx_sol < N; idx_sol++){
        //         printf("%i, ", sol[idx_sol]);
        //     }
        //     printf("] - Value: %i\n", d_sum[ant_id]);
        //     printf("------------------------------------------------\n");
        // }
        
        free(probs);
        free(choices);

    }
}

int main() {
    

    
    std::ifstream infile("bases_grafos/entrada1.txt");
    std::vector<std::vector<int>> adjList;
    int n1, n2, w;
    int N = 0;
    while (infile >> n1 >> n2 >> w)
    {
        if(n1 > N) N = n1;
        if(n2 > N) N = n2;
        adjList.push_back(std::vector<int>({n1-1, n2-1, w}));
    }

    std::cout << "Diff " <<  N << std::endl;

    int N_ITER = 100;
    int N_ANTS = 100;
    float EVAP = 0.2;
    
    
    // Pointers
    float *d_t;
    // float *t;
    int *sol, *sum;
    int *d_sol, *d_sum;
    int *d_visited;
    int *d_g, *g;
    int *best_sol;
    int best_sum = 0;

    // Host Array
    g = (int *)malloc(N * N * sizeof(int));
    // t = (float *)malloc(N * N * sizeof(float));

    sol = (int *)malloc(N * N_ANTS * sizeof(int));
    sum = (int *)malloc(N_ANTS * sizeof(int));
   
    best_sol = (int *)malloc(N * sizeof(int));
    
    // Device Array
    gpuErrchk(cudaMalloc(&d_t, N * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, N * N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_sol, N * N_ANTS * sizeof(int)));  // solutions
    gpuErrchk(cudaMalloc(&d_visited, N * N_ANTS * sizeof(int)));  // solutions
    gpuErrchk(cudaMalloc(&d_sum, N_ANTS * sizeof(int)));  // sums
    
    // Populate Graph
    for(auto it = std::begin(adjList); it != std::end(adjList); ++it) {
        int i = (*it)[0];
        int j = (*it)[1];
        int w = (*it)[2];
        g[(i*N)+j] = w;
    }

    gpuErrchk(cudaMemcpy(d_g, g, N*N*sizeof(int), cudaMemcpyHostToDevice));
    
    int nnBlocks = ((N*N) / THREADS_P_BLOCK) + 1;
    int nBlocks = (N / THREADS_P_BLOCK) + 1;

    // Setup Random Number Generator
    curandState_t* states;
    gpuErrchk(cudaMalloc((void**) &states, N_ANTS * sizeof(curandState_t)));
    rand_init<<<N_ANTS, 1>>>(time(0), states);
    
    
    
    for(int initial_node = 0; initial_node < N; initial_node ++){
        
        printf("Starting with node %i: \n", initial_node);
        
        reset_float<<<nnBlocks, THREADS_P_BLOCK>>>(d_t, N*N, 1.0);
        gpuErrchk( cudaDeviceSynchronize() );


        for(int iter = 0; iter < N_ITER; iter++){
            // printf("Iter...\n");
            
            // Reset Solutions, Visited list and Sum list
            reset_int<<<nnBlocks, THREADS_P_BLOCK>>>(d_sol, N * N_ANTS, -1);
            reset_int<<<nnBlocks, THREADS_P_BLOCK>>>(d_visited, N * N_ANTS, 0);
            reset_int<<<nBlocks, THREADS_P_BLOCK>>>(d_sum, N_ANTS, 0);
            gpuErrchk( cudaDeviceSynchronize() );
            
            // printf("start Ants\n");
            // Run Ants
            ant<<<nBlocks, THREADS_P_BLOCK>>>(states, d_t, d_g, N, N_ANTS, initial_node, d_sol, d_sum, d_visited);
            // ant<<<1, 1>>>(states, d_t, d_g, N, N_ANTS, initial_node, d_sol, d_sum, d_visited);
            gpuErrchk( cudaDeviceSynchronize() );
            // printf("End Ants\n");


            // Evaporate trail
            evaporate<<<nnBlocks, THREADS_P_BLOCK>>>(d_t, EVAP, N*N);
            gpuErrchk( cudaDeviceSynchronize() );
            
            // Update trail
            update_trail<<<nBlocks, THREADS_P_BLOCK>>>(d_t, N, N_ANTS, d_sol, d_sum);
            gpuErrchk( cudaDeviceSynchronize() );
            
            // //Print Trail
            // printmat<<<1, 1>>>(d_t, N);
            // gpuErrchk( cudaDeviceSynchronize() );

            // Pull solutions
            cudaMemcpy(sol, d_sol, N_ANTS*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(sum, d_sum, N_ANTS*sizeof(int), cudaMemcpyDeviceToHost);
            
            // Save Best Solution
            for(int i = 0; i < N_ANTS; i++){
                if(sum[i] > best_sum){
                    best_sum = sum[i];
                    memcpy(best_sol, &sol[i*N], N*sizeof(int));
                    printf("------------------------------------------------\n[");
                    for(int idx_sol = 0; idx_sol < N; idx_sol++){
                        printf("%i, ", best_sol[idx_sol]);
                    }
                    printf("] - Value: %i\n", best_sum);
                    printf("------------------------------------------------\n");

                }
                
            }
            
        }

        printf("Best Sol: %i\n", best_sum);
        
    }

    printf("%i\n", best_sum);

    cudaFree(d_sol);
    cudaFree(d_sum);
    cudaFree(d_t);
    cudaFree(d_visited);
    cudaFree(d_g);

    return 0;
}