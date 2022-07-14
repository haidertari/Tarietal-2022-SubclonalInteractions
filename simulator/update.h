#ifndef UPDATE_H
#define UPDATE_H


void Proliferate(int8_t *space, int8_t *E, int x, int y, int z, int agression, int (&cells)[2]){

  //get current cell type
  int idx = GetIDX(x,y,z);
  int8_t type = space[idx];
  if(type == 0){
    std::cout << "No cell selected to proliferate" << std::endl;
    return;
  }

  //find empty space to divide into
  int empty = E[idx];

  if(empty > 0){
    double choose = static_cast<double>(empty) * dis(gen);

    for(int i = -1; i <= 1; ++i){
      for(int j = -1; j <= 1; ++j){
        for(int k = -1; k <= 1; ++k){

          if(i == 0 && j == 0 && k == 0){
            continue;
          }

          if(space[GetIDX(x+i,y+j,z+k)] == 0){
            choose -= 1.0;
            if(choose <= 0.0){
              space[GetIDX(x+i, y+j, z+k)] = type;
              cells[type-1]++;
              return;
            }
          }
        }
      }
    }
    std::cout << "Overflow prolif neighbour" << std::endl;
  }

  else{

    //find a random direction to divide to
    int dx, dy, dz;
    do{
      dx = loc(gen);
      dy = loc(gen);
      dz = loc(gen);
    } while( dx == 0 && dy == 0 && dz == 0 );

    int N = 0;
    int xx = x;
    int yy = y;
    int zz = z;
    short ty;

    while( space[GetIDX(xx+dx, yy+dy, zz+dz)] != 0 && N < agression ){
      N += 1;
      xx += dx;
      yy += dy;
      zz += dz;
    }

    if( N == agression){
      return;
    }

    for(int i = N; i >= 0; --i){
      ty = space[GetIDX( x+(N*dx), y+(N*dy), z+(N*dz) )];
      space[GetIDX( x+((N+1)*dx), y+((N+1)*dy), z+((N+1)*dz))] = ty;
    }

    space[idx] = type;
    cells[type-1]++;
    return;
  }

}
void Move(int8_t *space, int8_t *E, int x, int y, int z){

  //get current cell type
  int idx = GetIDX(x,y,z);
  int8_t type = space[idx];
  if(type == 0){
    std::cout << "No cell selected to migrate" << std::endl;
    return;
  }

  int empty = E[idx];
  if(empty == 0){
    std::cout << "No space" << std::endl;
    return;
  }
  double choose = static_cast<double>(empty) * dis(gen);
  for(int i = -1; i <= 1; ++i){
    for(int j = -1; j <= 1; ++j){
      for(int k = -1; k <= 1; ++k){
        if(i == 0 && j == 0 && k == 0){
          continue;
        }
        if(space[GetIDX(x+i, y+j, z+k)] == 0){
          choose -= 1.0;
          if(choose <= 0.0){
            //swap
            space[idx] = 0;
            space[GetIDX(x+i, y+j, z+k)] = type;
            return;
          }
        }


      }
    }
  }
  std::cout << choose << " " << static_cast<int>(E[idx]) << std::endl;
  std::cout << "Overflow in move ..." << std::endl;
  exit(1);
}
void Death(int8_t *space, int x, int y, int z,  int (&cells)[2]){

  //get current cell type
  int idx = GetIDX(x,y,z);
  int8_t type = space[idx];
  if(type == 0){
    std::cout << "No cell selected to die" << std::endl;
    return;
  }

  space[idx] = 0;
  cells[type-1]--;
  return;

}
#endif
