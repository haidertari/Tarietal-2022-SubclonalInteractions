#ifndef INITIALISE_H
#define INITIALISE_H

//functions to initialise the array and calculate empty adjacent cells
__global__ void CalculateEmpty(int8_t* S, int8_t E);
__global__ void InitialiseSpace(int8_t *C,curandState* globalState,
                                int radius, double ratio);
//these functions set up a random number generator to use in cuda
__device__ double generate(curandState* globalState, int ind);
__global__ void setup_kernel ( curandState * state, unsigned long seed );
void SetRandom(curandState* devStates);





__global__ void InitialiseSpace(int8_t *C,curandState* globalState,
                                int radius, double ratio){
  //determine which cell is references
  int idx = blockIdx.x * blockDim.x + threadIdx.x;



  //calculate coordinates
  int x, y, z;
  d_CalculateXYZ(x,y,z,idx);

  //find distance to the cell
  int distance = pow((x-MID),2) + pow((y-MID),2) + pow((z-MID),2);

  double number = generate(globalState, idx);
  C[idx] = (number < ratio) ? 1:2;
  C[idx] = (distance < radius) ? C[idx]:0;

  return;

}
__global__ void CalculateEmpty(int8_t* S,int8_t* E){
    //calculate empty cells that can be migrated to or divided to

    //find the index of the cell
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //neighbour references assuming periodic boundary conditions
    __shared__ int a[3][26];
    if(threadIdx.x == 0){
      int temp[3][26] = {
        {-1,-1,-1,0,0,0,1,1,1,  -1,-1,-1,0,0,1,1,1,   -1,-1,-1,0,0,0,1,1,1},
        {-1,0,1,-1,0,1,-1,0,1,  -1,0,1,-1,1,-1,0,1,   -1,0,1,-1,0,1,-1,0,1},
        {1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,0,      -1,-1,-1,-1,-1,-1,-1,-1,-1}
      };
      for(int i = 0; i < 26; ++i){
        a[0][i] = temp[0][i];
        a[1][i] = temp[1][i];
        a[2][i] = temp[2][i];
      }
    }
    __syncthreads();

    //determine the coordinates
    int x,y,z;
    d_CalculateXYZ(x,y,z,idx);
    int dx,dy,dz;

    //counter for number of empty spaces
    int empty = 0;

    //ignore the empty voxels

    //find the empty cells for all occupied sites
    if(S[idx] != 0){
      for(int i = 0; i < 26; ++i){
        dx = x+a[0][i];
        dy = y+a[1][i];
        dz = z+a[2][i];
        if(S[d_GetIDX(dx,dy,dz)] == 0){
          empty++;
        }
      }
    }


    //write empty
    E[d_GetIDX(x,y,z)] = empty;
    return;
}
__device__ double generate(curandState* globalState, int ind){
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    double RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}
__global__ void setup_kernel ( curandState * state, unsigned long seed ){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
    return;
}
void SetRandom(curandState* devStates){
  srand(time(0));
  int seed = rand();
  setup_kernel<<<blocksforspace, threadsforspace>>>(devStates,seed);
  cudaDeviceSynchronize();
}

#endif
