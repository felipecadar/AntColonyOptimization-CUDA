#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <mutex>
#include <omp.h>

/* we need these includes for CUDA's random number stuff */
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define THREADS_P_BLOCK 128
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

__global__ void update_trail(float *t, int N, int N_ANTS, int N_EDGES, int* d_sol, int* sum){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N_ANTS; i += stride){
        int ant_id = i;
        int *sol = &d_sol[ant_id*N_EDGES];
        
        // for(int idx_sol = 0; idx_sol < N; idx_sol++){
        //     printf("%i ", sol[idx_sol]);
        // }

        // printf("\n");
        for(int idx_sol = 1; idx_sol < N_EDGES; idx_sol++){
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

__device__ int randChoice(curandState_t *state, float *prob, int N){
    float c = curand_uniform(state);
    float cum = 0;
    for(int i = 0; i < N; i++){
        if(c <= prob[i] + cum) return i;
        cum += prob[i];
    }
    return N-1;
}

__global__ void ant(curandState_t* states, float *t, int *g, int N, int N_ANTS,int N_EDGES, int init, int *d_sol, int *d_sum, int *d_visited, int alpha, int beta){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int ant_id = index; ant_id < N_ANTS; ant_id += stride){

        // int ant_id = i;
        int *visited = &(d_visited[ant_id*N]);
        int *sol = &(d_sol[ant_id*N_EDGES]);
        
        int now_node = init;
        int end = 0;
        int sol_idx = 0;
        
        float *probs = new float[N];
        // int *choices = new int[100];
        
        sol[sol_idx] = now_node;
        visited[now_node] = 1;
        d_sum[ant_id] = 0;

        
        while(end == 0){
            sol_idx++;
            // printf("\nNOW NODE: %i\nValid Neigs: ",now_node);
            //Calc Probs
            end = 1;
            float norm = 0;
            for(int neigh = 0; neigh < N; neigh ++){
                probs[neigh] = 0;
                if(g[now_node*N + neigh] > 0 && visited[neigh] == 0){
                    
                    float Tij = t[now_node*N + neigh];
                    float Nij = (float)g[now_node*N + neigh] / (float)MAX_WEIGHT;
                    
                    float res = pow(Tij, alpha) * pow(Nij, beta);
                    // printf("%i [%.2f, %.2f, %.2f], ", neigh, Tij, Nij, res);
                    
                    probs[neigh] = res;
                    norm += res;
                    
                    end = 0;
                }
            }

            //Norm probs to sum 1
            if(end) break;
            
            for(int neigh = 0; neigh < N; neigh ++){ 
                probs[neigh] = (probs[neigh] / norm);
            }


            int target = randChoice(&states[blockIdx.x], probs, N);
            assert(target >= 0 && target < N);

            d_sum[ant_id] += g[now_node*N + target];
            
            sol[sol_idx] = target;
            visited[target] = 1;

            // printf("SELECTED: %i\n", target);
            
            now_node = target;
            
        }

        // printf("END IN %i SOL LEN %i\n", now_node, sol_idx);
        // if(d_sum[ant_id] >= 177){
        //     printf("------------------------------------------------\n[");
        //     for(int idx_sol = 0; idx_sol < N; idx_sol++){
        //         printf("%i, ", sol[idx_sol]);
        //     }
        //     printf("] - Value: %i\n", d_sum[ant_id]);
        //     printf("------------------------------------------------\n");
        // }
        
        free(probs);
        // free(choices);

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
    if( argc > 6)
        exp_id = std::string(argv[7]);
        METRICS = 1;    

    
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
    std::mutex outfile_mutex;
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


    int nnBlocks = ((N*N) / THREADS_P_BLOCK) + 1;
    int nBlocks = (N / THREADS_P_BLOCK) + 1;

    int *best_do_best_sol;
    int *best_do_best_sum;

    best_do_best_sol = (int *)malloc(N_EDGES * N * sizeof(int));
    best_do_best_sum = (int *)malloc(N * sizeof(int));

    // Populate Graph
    int *g;
    g = (int *)malloc(N * N * sizeof(int));

    for(int i = 0; i < N*N; i++)
        g[i] = 0;

    for(auto it = std::begin(adjList); it != std::end(adjList); ++it) {
        int i = (*it)[0];
        int j = (*it)[1];
        int w = (*it)[2];
        g[(i*N)+j] = w;
    }
    
    int *d_g;
    gpuErrchk(cudaMalloc(&d_g, N * N * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_g, g, N*N*sizeof(int), cudaMemcpyHostToDevice));


    #pragma omp parallel for
    for(int initial_node = 0; initial_node < N; initial_node ++){
        int cpuid = omp_get_thread_num();
        cudaSetDevice(cpuid);
        printf("------- %i --------\n", cpuid);

        //create stream 
        cudaStream_t stream1;
        cudaStreamCreate(&stream1);

        // Pointers
        float *d_t;
        float *t;
        int *sol, *sum;
        int *d_sol, *d_sum;
        int *d_visited;
        int *best_sol;
        int best_sum = 0;

        // Host Array
        t = (float *)malloc(N * N * sizeof(float));
        sol = (int *)malloc(N_EDGES * N_ANTS * sizeof(int));
        sum = (int *)malloc(N_ANTS * sizeof(int));
        best_sol = (int *)malloc(N_EDGES * sizeof(int));
        
        // Device Array
        gpuErrchk(cudaMalloc(&d_t, N * N * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_sol, N_EDGES * N_ANTS * sizeof(int)));  // solutions
        gpuErrchk(cudaMalloc(&d_visited, N * N_ANTS * sizeof(int)));  // solutions
        gpuErrchk(cudaMalloc(&d_sum, N_ANTS * sizeof(int)));  // sums
        

        // Setup Random Number Generator
        curandState_t* states;
        gpuErrchk(cudaMalloc((void**) &states, nnBlocks * sizeof(curandState_t)));
        rand_init<<<nnBlocks, 1, 0, stream1>>>(time(0), states);
        
        printf("Starting with node %i: \n", initial_node);
        
        reset_float<<<nnBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_t, N*N, 1.0);
        gpuErrchk( cudaDeviceSynchronize() );
        

        for(int iter = 0; iter < N_ITER; iter++){
            // printf("Iter...\n");
            
            // Reset Solutions, Visited list and Sum list
            reset_int<<<nnBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_sol, N_EDGES * N_ANTS, -1);
            reset_int<<<nnBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_visited, N * N_ANTS, 0);
            reset_int<<<nBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_sum, N_ANTS, 0);
            gpuErrchk( cudaDeviceSynchronize() );
            
            // Run Ants
            // printf("Start Ants\n");
            ant<<<nBlocks, THREADS_P_BLOCK, 0, stream1>>>(states, d_t, d_g, N, N_ANTS, N_EDGES, initial_node, d_sol, d_sum, d_visited, alpha, beta);
            gpuErrchk( cudaDeviceSynchronize() );
            // printf("End Ants\n");
            
            
            // Evaporate trail
            evaporate<<<nnBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_t, EVAP, N*N);
            gpuErrchk( cudaDeviceSynchronize() );
            
            // Update trail
            update_trail<<<nBlocks, THREADS_P_BLOCK, 0, stream1>>>(d_t, N, N_ANTS, N_EDGES, d_sol, d_sum);
            
            // //Print Trail
            // printmat<<<1, 1, 0, stream1>>>(d_t, N);
            // gpuErrchk( cudaDeviceSynchronize() );
            
            // Pull solutions
            cudaMemcpyAsync(sol, d_sol, N_ANTS*N_EDGES*sizeof(int), cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(sum, d_sum, N_ANTS*sizeof(int), cudaMemcpyDeviceToHost, stream1);

            gpuErrchk( cudaDeviceSynchronize() );
            
            float mean_phero;
            if(METRICS){
                cudaMemcpyAsync(t, d_t, N*N*sizeof(float), cudaMemcpyDeviceToHost, stream1);
                mean_phero = 0;
                for(int i = 0; i < N*N; i++){
                    mean_phero += t[i];
                }

                mean_phero = mean_phero / (float)N*N;
            }

            if(METRICS)
                outfile_mutex.lock();

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

            if(METRICS)
                outfile_mutex.unlock();
            
        }

        // printf("------------------------------------------------\n[");
        // for(int idx_sol = 0; idx_sol < N_EDGES; idx_sol++){
        //     if(best_sol[idx_sol] == -1) break;
        //     printf("%i, ", best_sol[idx_sol]);
        // }
        // printf("]\n");
        // printf("------------------------------------------------\n");
        // printf("%i\n", best_sum);

        memcpy(&best_do_best_sol[initial_node * N_EDGES], best_sol, N_EDGES*sizeof(int));
        best_do_best_sum[initial_node] = best_sum;        

        printf("[End from node %i] - Best Sol: %i\n", initial_node, best_sum);

        cudaFree(d_sol);
        cudaFree(d_sum);
        cudaFree(d_t);
        cudaFree(d_visited);

        free(sol);
        free(sum);
        free(best_sol);

        //end stream
        cudaStreamDestroy(stream1);
    }

    // Get best general solution
    int *best_sol;
    int best_sum = 0;

    best_sol = (int *)malloc(N_EDGES * sizeof(int));

    for(int i = 0; i < N; i++){
        if (best_do_best_sum[i] > best_sum){
            best_sum = best_do_best_sum[i];
            memcpy(best_sol, &best_do_best_sol[i*N_EDGES], N_EDGES*sizeof(int));                    
        }
    }
    

    printf("------------------------------------------------\n[");
    for(int idx_sol = 0; idx_sol < N_EDGES; idx_sol++){
        if(best_sol[idx_sol] == -1) break;
        printf("%i, ", best_sol[idx_sol]);
    }
    printf("]\n");
    printf("------------------------------------------------\n");
    printf("%i\n", best_sum);
    
    if(METRICS){
        for(int idx_sol = 0; idx_sol < N_EDGES; idx_sol++){
            if(best_sol[idx_sol] == -1) break;
            outfile << best_sol[idx_sol] << " ";
        }
        outfile << std::endl;
        outfile << best_sum << std::endl;
    }

    int s = 0;
    for(int idx_sol = 1; idx_sol < N_EDGES; idx_sol++){
        int from = best_sol[idx_sol - 1 ];
        int to = best_sol[idx_sol];
        if(to == -1) break;
        s += g[from * N + to];
    }

    if(s != best_sum){
        printf("SOLUTION DO NOT MATCH VALUE\n");
        printf("%i vs %i\n", s, best_sum);
    }
    


    if(METRICS)
        outfile.close();

    free(best_do_best_sol);
    free(best_do_best_sum);
    free(best_sol);
    free(g);
    cudaFree(d_g);

    return 0;
}
