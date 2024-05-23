//
// Created by Renzo on 23-6-2020.
//

#ifndef AMALGAM_RANDOM_H
#define AMALGAM_RANDOM_H

#include <math.h>

double randomRealUniform01( void );
int randomInt( int maximum );
double random1DNormalUnit( void );
double random1DNormalParameterized( double mean, double variance );

#endif //AMALGAM_RANDOM_H
