//
// Created by Renzo on 13-6-2020.
//

#ifndef PROJECT_PROBLEMS_H
#define PROJECT_PROBLEMS_H

#include <stdlib.h>
#include <math.h>
#include "shared/matrix.h"

char *installedProblemName( int index );
int numberOfInstalledProblems( void );
double installedProblemLowerRangeBound( int index );
double installedProblemUpperRangeBound( int index );
short isParameterInRangeBounds( int index, double parameter );
void installedProblemEvaluation( int index, int number_of_parameters, double* parameters, double* objective_value, double* constraint_value, int& number_of_evaluations );
void sphereFunctionProblemEvaluation( int number_of_parameters, double* parameters, double* objective_value, double* constraint_value );
void ciasFunctionProblemEvaluation(int number_of_parameters, double *parameters, double *objective_value, double *constraint_value );
double sphereFunctionLowerRangeBound( void );
double sphereFunctionUpperRangeBound( void );
double ciasFunctionLowerRangeBound( void );
double ciasFunctionUpperRangeBound( void );

#endif //PROJECT_PROBLEMS_H
