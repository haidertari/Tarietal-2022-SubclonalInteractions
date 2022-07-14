#ifndef SPACE_H
#define SPACE_H


#include "indexing.h"
#include "initialise.h"
#include "propensity.h"
#include "update.h"


class Space{
private:
  int8_t *d_space, *h_space, *d_empty, *h_empty, reaction;
  double *d_pro, *h_pro, *d_mig, *h_mig, *d_dea, *h_dea, *d_totalmig;
  double *d_totalpro, *d_totaldea, *d_rates;
  double sum, t, rates[numrec], *h_totalpro, *h_totaldea, *h_totalmig;
  int ridx,x,y,z,cells[2];
  int8_t *condense1, *condense2;
public:
  Space(const double (&pro)[2], const double (&mig)[2], const double (&dea)[2],
    const double (&inter)[2], double radius, double ratio){
    //set time to zero
    t = 0.0;
    //copy rates
    rates[0] = pro[0];rates[1] = pro[1];rates[2] = mig[0];
    rates[3] = mig[1];rates[4] = dea[0];rates[5] = dea[1];
    rates[6] = inter[0]; rates[7] = inter[1];
    //allocate all cuda variables
    cudaMalloc(&d_rates,sizeof(double)*numrec);
    cudaMalloc(&d_space,spacebytes); //cells array
    cudaMalloc(&d_empty,spacebytes); //empty space for each cell
    cudaMalloc(&d_pro,propbytes); //proliferation propensity
    cudaMalloc(&d_mig,propbytes); //migration propensity
    cudaMalloc(&d_dea,propbytes); //migration propensity
    cudaMalloc(&d_totalpro,sizeof(double));
    cudaMalloc(&d_totaldea,sizeof(double));
    cudaMalloc(&d_totalmig,sizeof(double));
    //allocate host variables
    condense1 = (int8_t*)malloc(sizeof(int8_t)*DIM*DIM);
    condense2 = (int8_t*)malloc(sizeof(int8_t)*DIM*DIM);

    h_space = (int8_t*)malloc(spacebytes);
    h_empty = (int8_t*)malloc(spacebytes);
    h_mig = (double*)malloc(propbytes);
    h_pro = (double*)malloc(propbytes);
    h_dea = (double*)malloc(propbytes);
    h_totalmig = (double*)malloc(sizeof(double));
    h_totalpro = (double*)malloc(sizeof(double));
    h_totaldea = (double*)malloc(sizeof(double));

    //temporarily create random numbers
    curandState* devStates;
    cudaMalloc(&devStates,DIM*DIM*DIM*sizeof(curandState));
    SetRandom(devStates);

    //initialise the space array
    InitialiseSpace<<<blocksforspace,threadsforspace>>>(d_space,devStates,
      radius,ratio);
    cudaFree(devStates);
    cudaDeviceSynchronize();


    //copy to host/device
    cudaMemcpy(h_space,d_space,spacebytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(d_rates,rates,sizeof(double)*numrec,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //calculate total cells
    cells[0] = 0;
    cells[1] = 0;
    for(int i = 0; i < DIM*DIM*DIM; ++i){
      if(h_space[i] == 1){
        cells[0] += 1;
      }
      else if(h_space[i] == 2){
        cells[1] += 1;
      }
    }

  };
  ~Space(){
    cudaFree(d_space);
    cudaFree(d_empty);
    cudaFree(d_totalmig);
    cudaFree(d_totalpro);
    cudaFree(d_totaldea);
    cudaFree(d_mig);
    cudaFree(d_pro);
    cudaFree(d_dea);
    cudaFree(d_rates);
    free(h_space);
    free(h_empty);
    free(h_mig);
    free(h_pro);
    free(h_dea);
    free(h_totaldea);
    free(h_totalmig);
    free(h_totalpro);
    free(condense1);
    free(condense2);
  };
  void UpdateEmpty(){
    CalculateEmpty<<<blocksforspace,threadsforspace>>>(d_space,d_empty);
    cudaDeviceSynchronize();
    cudaMemcpy(h_empty,d_empty,spacebytes,cudaMemcpyDeviceToHost);
    return;
  };
  void Propensity(){
    double prop = static_cast<double>(cells[0])/static_cast<double>(cells[0]+cells[1]);
    //calculate propensity per point
    CalculatePropensity<<<blocksforspace,threadsforspace>>>(d_pro,d_mig,d_dea,
                                                      d_space,d_empty,d_rates,prop);
    cudaDeviceSynchronize();

    //sum propensity
    *h_totaldea = 0.0;
    *h_totalmig = 0.0;
    *h_totalpro = 0.0;
    cudaMemcpy(d_totalpro,h_totalpro,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_totalmig,h_totalmig,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_totaldea,h_totaldea,sizeof(double),cudaMemcpyHostToDevice);
    SumPropensity<<<blocksforsum,threadsforsum>>>(d_mig,DIM*DIM,d_totalmig);
    SumPropensity<<<blocksforsum,threadsforsum>>>(d_pro,DIM*DIM,d_totalpro);
    SumPropensity<<<blocksforsum,threadsforsum>>>(d_dea,DIM*DIM,d_totaldea);
    cudaDeviceSynchronize();
    //copy to host
    cudaMemcpy(h_pro,d_pro,propbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mig,d_mig,propbytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dea,d_dea,propbytes,cudaMemcpyDeviceToHost);


    cudaMemcpy(h_totalpro,d_totalpro,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_totalmig,d_totalmig,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_totaldea,d_totaldea,sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return;
  };
  void FindReactionandTime(){

      //find total pro
      double total[3] = {*h_totalpro,*h_totalmig,*h_totaldea};
      sum = total[0]+total[1]+total[2];

      //update by time to next reaction
      t += (1.0/sum) * log(1.0/dis(gen));

      //find which reaction happens
      double choose = dis(gen) * sum;
      for(int i = 0; i < 3; ++i){
        choose -= total[i];
        if(choose < 0.00001){
          reaction = i+1;
          break;
        }
        if(i == 3){
          std::cout << "choosing reaction overflow" << std::endl;
        }
      }
      choose = dis(gen) * total[reaction-1];
      switch(reaction){
        case 1:
          for(int i = 0; i < DIM*DIM*DIM; ++i){
            choose -= h_pro[i];

            if(choose < 0.00001){
              //std::cout << "pro: "  << h_pro[i] << " " << choose << " " << static_cast<int>(h_space[i])<< std::endl;
              ridx = i;
              break;
            }
          }
        break;
        case 2:
          for(int i = 0; i < DIM*DIM*DIM; ++i){
            choose -= h_mig[i];
            if(choose < 0.00001){
              ridx = i;
              //std::cout << "mig: " << h_mig[i] << " " << choose << " " << static_cast<int>(h_space[i])<< std::endl;
              break;
            }
          }
        break;
        case 3:
          for(int i = 0; i < DIM*DIM*DIM; ++i){
            choose -= h_dea[i];
            if(choose < 0.00001){
              //std::cout << "dea: "  << h_dea[i] << " " << choose << " " << static_cast<int>(h_space[i])<< std::endl;
              ridx = i;
              break;
            }
          }
          break;
      }

      CalculateXYZ(x,y,z,ridx);
      return;
  };
  void ExecuteReaction(int agression){
    switch(reaction){
        case 1:
          Proliferate(h_space, h_empty, x, y, z, agression, cells);
          break;
        case 2:
          Move(h_space, h_empty, x, y, z);
          break;
        case 3:
          Death(h_space, x, y, z, cells);
          break;
        default:
          std::cout << "No reaction ..." << std::endl;
          exit(1);
      }
      //update to the new state
      cudaMemcpy(d_space,h_space,spacebytes,cudaMemcpyHostToDevice);

  }
  double GetTime(){
    return t;
  };
  void CondenseArray(){
    int tmp;
    for(int i = 0; i < DIM*DIM; ++i){
      condense1[i] = 0;
      condense2[i] = 0;
     }
    for(int xx = 0; xx < DIM; ++xx){
      for(int yy = 0; yy < DIM; ++yy){
        for(int zz = 0; zz < DIM; ++zz){
          tmp = GetIDX(xx,yy,zz);
          if(h_space[tmp] != 0){
            if(h_space[tmp] == 1){
              condense1[xx + (yy*DIM)] = 1;
            }
            else if(h_space[tmp] == 2){
              condense2[xx + (yy*DIM)] = 1;
            }
          }
        }
      }
    }
    return;
  }
  void PrintSpace(){
    CondenseArray();
    std::cout << std::endl;
    for(int i = 0; i < DIM*DIM; ++i){
      if(i % DIM == 0){
        std::cout << std::endl;
      }
      std::cout << static_cast<int>(condense2[i]);
    }
    return;
  };
  void WriteState(std::ofstream &file,int8_t id,double ratio){
    CondenseArray();
    file.write((char*) &id, sizeof(int8_t));
    file.write((char*) &rates, sizeof(double)*numrec);
    file.write((char*) &ratio, sizeof(double));
    file.write((char*) &t, sizeof(t));
    file.write((char*) condense1, DIM*DIM*sizeof(int8_t));
    file.write((char*) condense2,  DIM*DIM*sizeof(int8_t));
    return;
  }
  int Cell(int i, int j, int k){
    return static_cast<int>(h_space[GetIDX(i,j,k)]);
  }
};




#endif
