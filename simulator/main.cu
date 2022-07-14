#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <string>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

//Global parameters for determination of space and cuda block and thread
const int DIM = 64;
const int MID = 32;
const int numrec = 8;
const int spacebytes = sizeof(int8_t) * DIM * DIM * DIM;
const int propbytes = sizeof(double) * DIM * DIM * DIM;
dim3 blocksforspace( DIM*DIM,1,1);
dim3 threadsforspace(DIM,1,1);
dim3 blocksforsum(DIM*DIM,1,1);
dim3 threadsforsum(DIM,1,1);

//Random devices, seed and distribution (can be changed based on need)
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0,1);
std::uniform_int_distribution<int> loc(-1,1);
std::uniform_real_distribution<double> rpro(0,0.5);
std::uniform_real_distribution<double> rmig(0,8);
//std::uniform_real_distribution<double> rinter(0,8);

//insert priors for rates here if these have been recovered
//std::normal_distribution<double> rmig(0.1,1.2);


#include "runsimulation.h"

//Setting precision of numbers by rounding
double Round(double number,double n){
  number = number * n;
  number = round(number)/n;
  return number;
}

//Set up simulation to be run (this can be changed based on need)
void SetUpSim(std::string s, int sid){

  std::ofstream file;
  file.open(s,std::ios::binary);

  double pro[2] = {Round(0.05,100),Round(0.3,100)};
  double dea[2] = {Round(0.005,1000),Round(0.03,1000)};
  double mig[2] = {Round(0.5,100),Round(7.5,100)};
  double inter[2] = {Round(0.0,10),Round(0.0,10)};

  double radius = 36.1;
  double agression = 1;
  double maxtime = 3.01;
  double ratio;

  for(double rat = 0.0; rat <= 1.0; rat+=0.2){

    pro[0] = Round(rpro(gen),100);
    pro[1] = Round(rpro(gen),100);
    mig[0] = Round(rpro(gen),100);
    mig[1] = Round(rpro(gen),100);
    dea[0] = Round(rpro(gen),100);
    dea[1] = Round(rpro(gen),100);
    inter[0] = 0;
    inter[1] = 0;
    ratio = Round(rat,1000);

    RunSimulation(pro,mig,dea,inter,radius,ratio,agression,maxtime,sid,file);

  }
  file.close();
  return;
}


int main(){
  int start = 0;

  std::vector<std::thread> vec_thr;
  std::string s;


  for(int i = start; i < start+10; ++i){
    s = "Test/Test-Sim"+std::to_string(i);
    vec_thr.push_back(std::thread(SetUpSim,s,i));
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  for(int i = 0; i < 10; ++i){
    vec_thr[i].join();
  }

  return 0;
}
