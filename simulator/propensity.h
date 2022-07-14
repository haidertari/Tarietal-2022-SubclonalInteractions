#ifndef PROPENSITY_H
#define PROPENSITY_H

__device__ double atomicadd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void SumPropensity(double *g_idata,int N, double *g_odata) {

    __shared__ double sdata[DIM*DIM];


    int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = g_idata[i];

    __syncthreads();

    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }


    if (threadIdx.x == 0)
        atomicadd(g_odata,sdata[0]);
}

__global__ void CalculatePropensity(double *pro, double *mig, double* dea, int8_t *S,
  int8_t *E, double *rates, double prop){

    //determine which voxel we are referencing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double cellprop[2];
    if(threadIdx.x == 0){
      cellprop[0] = 1.0 - prop;
      cellprop[1] = prop;
    }
    __syncthreads();

    //determine the x,y,z coordinates of the index
    int type, empty;
    //determine the type and the empty spaces next to the cell
    type = S[idx];
    empty = E[idx];
    pro[idx] = 0.0;
    dea[idx] = 0.0;
    mig[idx] = 0.0;

    pro[idx] = (type == 0) ? 0.0 : rates[type-1];
    dea[idx] = (type == 0) ? 0.0 : rates[type+3];
    mig[idx] = (type == 0) ? 0.0 : ( (rates[type+5] * cellprop[type-1]) + rates[type+1] );
    mig[idx] = (empty == 0 || mig[idx] <= 0.0) ? 0.0 : mig[idx];

    __syncthreads();
    return;

}

#endif
