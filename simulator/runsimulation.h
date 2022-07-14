#ifndef RUNSIMULATION_H
#define RUNSIMULATION_H

#include "space.h"

void RunSimulation(const double (&pro)[2], const double (&mig)[2],
                   const double (&dea)[2], const double (&inter)[2],
                   double radius, double ratio, double agression,
                   double maxtime, int8_t sID, std::ofstream &file)
{
  //Declare space
  Space space(pro, mig, dea, inter, radius, ratio);
  double printtime = 0.0;
  while(space.GetTime() < maxtime){

    //update empty space
    space.UpdateEmpty();

    //calculate propensity
    space.Propensity();
    //find reaction and time

    space.FindReactionandTime();

    //execute reaction (including copying the state to device)
    space.ExecuteReaction(agression);

    //write state
    if(printtime <= space.GetTime()){
      space.WriteState(file,sID,ratio);
      printtime += 1.0;

      if(printtime > maxtime){
        return;
      }
    }
  }

  return;
}

#endif
