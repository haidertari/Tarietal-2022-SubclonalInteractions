#ifndef INDEXING_H
#define INDEXING_H


//these two function translate 1D to 3D array (both ways) on device
__device__ void d_CalculateXYZ(int &x, int &y, int &z, int idx);
__device__ int d_GetIDX(int x, int y, int z);

//these two function translate 1D to 3D array (both ways) on CPU
int GetIDX(int x, int y, int z);
void CalculateXYZ(int &x, int &y, int &z, int idx);




int GetIDX(int x, int y, int z){

  x = (x+DIM)%DIM;
  y = (y+DIM)%DIM;
  z = (z+DIM)%DIM;
  int index = x + (y * DIM) + (z * DIM * DIM);
  return index;

}

void CalculateXYZ(int &x, int &y, int &z, int idx){

  x = idx % DIM;
  y = ((idx-x)/DIM) % DIM;
  z = ((idx - y*DIM - x)/(DIM*DIM))%DIM;
  return;

}




__device__ void d_CalculateXYZ(int &x, int &y, int &z, int idx){
  x = idx % DIM;
  y = ((idx-x)/DIM) % DIM;
  z = ((idx - y*DIM - x)/(DIM*DIM))%DIM;
  return;
}
__device__ int d_GetIDX(int x, int y, int z){

  x = (x+DIM)%DIM;
  y = (y+DIM)%DIM;
  z = (z+DIM)%DIM;
  int xyz = x + (y * DIM) + (z * DIM * DIM);
  return xyz;

}


#endif
