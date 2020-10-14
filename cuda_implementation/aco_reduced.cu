#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <chrono>


/* we need these includes for CUDA's random number stuff */
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define THREADS_P_BLOCK 1024
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


// Reset a FLOAT array in GPU
__global__ void reset_float(float *v, long int N, float val){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        v[i] = val;
}

// Reset a INT array in GPU
__global__ void reset_int(int *v, int N, int val){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        v[i] = val;
    }
}

// Evaporate pheromone matrix in GPU
__global__ void evaporate(float *t, float p, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        t[i] = t[i] * (1-p);
        if(t[i] < 1) t[i] = 1;
    }
}

// Prinf stuff in GPU without copy it to host
// use <<<1,1>>> or <<<1,1,0,stream>>> 
// More blocks and threads will print things
// in wrong order
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

// Update trail in GPU 
__global__ void update_trail(float *t, int N, int N_ANTS, int N_EDGES, int* d_sol, int* sum){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N_ANTS; i += stride){
        int ant_id = i;
        int *sol = &d_sol[ant_id*N_EDGES]; // Find solution from ant I
        
        // For each node in solution, update trail
        for(int idx_sol = 1; idx_sol < N_EDGES; idx_sol++){ 
            if(sol[idx_sol] == -1){
                break;
            }
            int from = sol[idx_sol-1];
            int to = sol[idx_sol];
            t[from*N + to] += sum[ant_id];
        }
    }
}

// Select a random int from 0 to N-1
// with probability in array prob, of size N
__device__ int randChoice(curandState_t *state, float *prob, int N){
    float c = curand_uniform(state);
    float cum = 0;
    for(int i = 0; i < N; i++){
        if(c <= prob[i] + cum) return i;
        cum += prob[i];
    }
    return N-1;
}

// Run the routine for one or more ants, depends on 
// the number of threads and blocks avalilable
__global__ void ant(curandState_t* states, float *t, int *g, int N, int N_ANTS,int N_EDGES, int *d_sol, int *d_sum, int *d_visited, int alpha, int beta){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int ant_id = index; ant_id < N_ANTS; ant_id += stride){

        // int ant_id = i;

        // Get pointers to array corresponding to this Ant
        int *visited = &(d_visited[ant_id*N]);
        int *sol = &(d_sol[ant_id*N_EDGES]);
        
        // Random Initial node
        float c = curand_uniform(&states[blockIdx.x]);
        int now_node = (int)(c * N);

        // Init aux variables
        int end = 0;
        int sol_idx = 0;
        
        // Alloc probability array to select next node
        float *probs = new float[N];
        
        // Init solution
        sol[sol_idx] = now_node;
        visited[now_node] = 1;
        d_sum[ant_id] = 0;

        
        while(end == 0){

            sol_idx++;

            //Calc Neighs Probs
            end = 1;
            float norm = 0;
            for(int neigh = 0; neigh < N; neigh ++){
                probs[neigh] = 0;

                // If has valid neigh
                if(g[now_node*N + neigh] > 0 && visited[neigh] == 0){
                    
                    float Tij = t[now_node*N + neigh];
                    float Nij = (float)g[now_node*N + neigh] / (float)MAX_WEIGHT;
                    
                    float res = pow(Tij, alpha) * pow(Nij, beta);
                    
                    probs[neigh] = res;
                    norm += res;
                    
                    end = 0;
                }
            }

            // Have no valid neighs
            if(end) break;
            
            //Norm probs to sum 1
            for(int neigh = 0; neigh < N; neigh ++){ 
                probs[neigh] = (probs[neigh] / norm);
            }

            // Choose next node
            int target = randChoice(&states[blockIdx.x], probs, N);
            assert(target >= 0 && target < N);

            // Add target to solution
            d_sum[ant_id] += g[now_node*N + target];
            sol[sol_idx] = target;
            visited[target] = 1;
            
            // Move Ant
            now_node = target;

        }

        // Free stuff
        free(probs);
    }
}

void printHelp(){
    std::cout << std::endl;
    std::cout << "Usage: ./ACO <input database> <N_ITER> <N_ANTS> <EVAPORATION RATE> <ALPHA> <BETA>" << std::endl;
    exit(0);
}

int main(int argc, char* argv[]) {

    if( argc < 6) printHelp();
    
    std::string database(argv[1]);
    
    
    std::ifstream infile(database);
    std::vector<std::vector<int>> adjList;
    
    int N = 0;
    int N_EDGES = 0;
    int N_ITER = atoi(argv[2]);
    int N_ANTS = atoi(argv[3]);
    float EVAP = atof(argv[4]);
    int alpha = atoi(argv[5]);
    int beta = atoi(argv[6]);
    
    int METRICS = 0;
    std::string exp_id;
    if( argc > 7){
        exp_id = std::string(argv[7]);
        METRICS = 1;    
    }

    
    int n1, n2, w;
    while (infile >> n1 >> n2 >> w)
    {
        N_EDGES++;
        if(n1 > N) N = n1;
        if(n2 > N) N = n2;
        adjList.push_back(std::vector<int>({n1-1, n2-1, w}));
    }

    infile.close();

    std::cout << "--------------- Config ---------------" << std::endl;
    std::cout << "Database:         " << database << std::endl;
    std::cout << "N Vertex:         " << N << std::endl;
    std::cout << "N Edges:          " << N_EDGES << std::endl;
    std::cout << "N Ants:           " << N_ANTS  << std::endl;
    std::cout << "Max Iterations:   " << N_ITER  << std::endl;
    std::cout << "Evaportation:     " << EVAP  << std::endl;
    std::cout << "alpha:            " << alpha << std::endl; 
    std::cout << "beta:             " << beta  << std::endl;
    std::cout << "Exp:              " << exp_id  << std::endl;
    std::cout << "--------------------------------------" << std::endl << std::endl;
    
    std::ofstream outfile;
    if(METRICS){
        outfile.open("results/" + exp_id + ".txt");
        outfile << "DATABASE " << database << std::endl;
        outfile << "N " << N << std::endl;
        outfile << "N_EDGES " << N_EDGES << std::endl;
        outfile << "N_ANTS " << N_ANTS  << std::endl;
        outfile << "N_ITER " << N_ITER  << std::endl;
        outfile << "EVAP " << EVAP  << std::endl;
        outfile << "alpha " << alpha << std::endl; 
        outfile << "beta " << beta  << std::endl;
    }
    
    // Pointers
    float *d_t;
    float *t;
    int *sol, *sum;
    int *d_sol, *d_sum;
    int *d_visited;
    int *d_g, *g;
    int *best_sol;
    int best_sum = 0;


    // Host Array
    g = (int *)malloc(N * N * sizeof(int));
    t = (float *)malloc(N * N * sizeof(float));

    sol = (int *)malloc(N_EDGES * N_ANTS * sizeof(int));
    sum = (int *)malloc(N_ANTS * sizeof(int));
   
    best_sol = (int *)malloc(N_EDGES * sizeof(int));
    
    // Device Array
    gpuErrchk(cudaMalloc(&d_t, N * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_g, N * N * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_sol, N_EDGES * N_ANTS * sizeof(int)));  // solutions
    gpuErrchk(cudaMalloc(&d_visited, N * N_ANTS * sizeof(int)));  // solutions
    gpuErrchk(cudaMalloc(&d_sum, N_ANTS * sizeof(int)));  // sums
    
    // Populate Graph
    for(int i = 0; i < N*N; i++)
        g[i] = 0;

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
    gpuErrchk(cudaMalloc((void**) &states, nnBlocks * sizeof(curandState_t)));
    rand_init<<<nnBlocks, 1>>>(time(0), states);
  
    
    int initial_node = 0;
        
    reset_float<<<nnBlocks, THREADS_P_BLOCK>>>(d_t, N*N, 1.0);
    gpuErrchk( cudaDeviceSynchronize() );

    
    for(int iter = 0; iter < N_ITER; iter++){
        // printf("Iter...\n");
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        
        // Reset Solutions, Visited list and Sum list
        reset_int<<<nnBlocks, THREADS_P_BLOCK>>>(d_sol, N_EDGES * N_ANTS, -1);
        reset_int<<<nnBlocks, THREADS_P_BLOCK>>>(d_visited, N * N_ANTS, 0);
        reset_int<<<nBlocks, THREADS_P_BLOCK>>>(d_sum, N_ANTS, 0);
        gpuErrchk( cudaDeviceSynchronize() );
        
        // Run Ants
        // printf("Start Ants\n");
        ant<<<nBlocks, THREADS_P_BLOCK>>>(states, d_t, d_g, N, N_ANTS, N_EDGES, d_sol, d_sum, d_visited, alpha, beta);
        gpuErrchk( cudaDeviceSynchronize() );
        // printf("End Ants\n");
        
        
        // Evaporate trail
        evaporate<<<nnBlocks, THREADS_P_BLOCK>>>(d_t, EVAP, N*N);
        gpuErrchk( cudaDeviceSynchronize() );
        
        // Update trail
        update_trail<<<nBlocks, THREADS_P_BLOCK>>>(d_t, N, N_ANTS, N_EDGES, d_sol, d_sum);
        
        // //Print Trail
        // printmat<<<1, 1>>>(d_t, N);
        // gpuErrchk( cudaDeviceSynchronize() );
        
        // Pull solutions
        cudaMemcpy(sol, d_sol, N_ANTS*N_EDGES*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sum, d_sum, N_ANTS*sizeof(int), cudaMemcpyDeviceToHost);

        gpuErrchk( cudaDeviceSynchronize() );
        

        float mean_phero;
        if(METRICS){
            cudaMemcpy(t, d_t, N*N*sizeof(float), cudaMemcpyDeviceToHost);
            mean_phero = 0;
            for(int i = 0; i < N*N; i++){
                mean_phero += t[i];
            }

            mean_phero = mean_phero / (float)N*N;
        }

        if(METRICS){
            outfile << "START_NODE " << initial_node << " ITER " << iter << " MEAN_PHERO " << mean_phero << " : ";
        }
        
        // Save Best Solution
        for(int i = 0; i < N_ANTS; i++){
            if(sum[i] > best_sum){
                best_sum = sum[i];
                memcpy(best_sol, &sol[i*N_EDGES], N_EDGES*sizeof(int));                    
            }

            if(METRICS){
                outfile << sum[i] << " ";
            }
            
        }
        if(METRICS){
            outfile << std::endl;
        }


        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "[" << iter << "] " << "Best sum: " << best_sum << " - Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        
    }
    
    printf("Best Sol: %i\n", best_sum);


    printf("------------------------------------------------\n[");
    for(int idx_sol = 0; idx_sol < N_EDGES; idx_sol++){
        if(best_sol[idx_sol] == -1) break;
        printf("%i, ", best_sol[idx_sol]+1);
    }
    printf("]\n");
    printf("------------------------------------------------\n");
    printf("%i\n", best_sum);
    
    if(METRICS){
        for(int idx_sol = 0; idx_sol < N_EDGES; idx_sol++){
            if(best_sol[idx_sol] == -1) break;
            outfile << best_sol[idx_sol]+1 << " ";
        }
        outfile << std::endl;
        outfile << best_sum << std::endl;
    }

    std::cout << "Checking Solution..." << std::endl;
    int s = 0;
    for(int idx_sol = 1; idx_sol < N_EDGES; idx_sol++){
        int from = best_sol[idx_sol - 1 ];
        int to = best_sol[idx_sol];
        if(to == -1) break;
        s += g[from * N + to];
        printf("G[%i -> %i] : %i | Sum: %i\n", from, to, g[from * N + to], s);
    }
    
    if(s != best_sum){
        printf("SOLUTION DO NOT MATCH VALUE\n");
        printf("%i vs %i\n", s, best_sum);
    }
    std::cout << "Done!" << std::endl;
    
    cudaFree(d_sol);
    cudaFree(d_sum);
    cudaFree(d_t);
    cudaFree(d_visited);
    cudaFree(d_g);

    if(METRICS)
        outfile.close();

    return 0;
}