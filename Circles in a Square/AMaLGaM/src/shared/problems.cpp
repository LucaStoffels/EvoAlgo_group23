//
// Created by Renzo on 13-6-2020.
//

#include "shared/problems.h"

/**
 * Returns the name of an installed problem.
 */
char *installedProblemName( int index )
{
  switch( index )
  {
    case  0: return((char*)"Sphere" );
    case  1: return((char*)"Circles in a Square");
  }
  
  return( NULL );
}

/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems( void )
{
  static int result = -1;
  
  if( result == -1 )
  {
    result = 0;
    while( installedProblemName( result ) != NULL )
      result++;
  }
  
  return( result );
}

/**
 * Returns the lower-range bound of an installed problem.
 */
double installedProblemLowerRangeBound( int index )
{
  switch( index )
  {
    case  0: return(sphereFunctionLowerRangeBound());
    case  1: return(ciasFunctionLowerRangeBound());
  }
  
  return( 0.0 );
}

/**
 * Returns the upper-range bound of an installed problem.
 */
double installedProblemUpperRangeBound( int index )
{
    switch (index)
    {
        case  0: return(sphereFunctionUpperRangeBound());
        case  1: return(ciasFunctionUpperRangeBound());
    }

    return(0.0);
}

/**
 * Returns whether a parameter is inside the range bound of
 * every problem.
 */
short isParameterInRangeBounds( int index, double parameter )
{
    if (parameter > installedProblemUpperRangeBound(index) ||
        parameter < installedProblemLowerRangeBound(index) ||
        isnan(parameter))
    {
        return(0);
    }
  
    return( 1 );
}

/**
 * Returns the value of the single objective and
 * the sum of all constraint violations function.
 * Both are returned using pointer variables.
 */
void installedProblemEvaluation( int index, int number_of_parameters, double *parameters, double *objective_value, double *constraint_value, int & number_of_evaluations )
{
    //  double *rotated_parameters;

    *objective_value = 0.0;
    *constraint_value = 0.0;

    number_of_evaluations++;
    switch (index)
    {
        case  0: sphereFunctionProblemEvaluation(number_of_parameters, parameters, objective_value, constraint_value); break;
        case  1: ciasFunctionProblemEvaluation(number_of_parameters, parameters, objective_value, constraint_value); break;
    }
}

void sphereFunctionProblemEvaluation( int number_of_parameters, double* parameters, double* objective_value, double* constraint_value )
{
    int i;
    double result = 0.0;
    for (i = 0; i < number_of_parameters; i ++) {
        result += pow(parameters[i], 2.0);
    }

    *objective_value = -result;
    *constraint_value = 0;
}

void ciasFunctionProblemEvaluation( int number_of_parameters, double *parameters, double *objective_value, double *constraint_value )
{
    int i, j;
    double result = 1000.0;
    for (i = 0; i < number_of_parameters; i += 2) {
        // Check distance with other points, take the smallest distance
        for (j = i + 2; j < number_of_parameters; j += 2) {
            double dist = sqrt(pow((parameters[i] - parameters[j]), 2.0) + pow((parameters[i + 1] - parameters[j + 1]), 2.0));
            if (dist < result) {
                result = dist;
            }
        }
    }

    *objective_value = result;
    *constraint_value = 0;
}


double sphereFunctionLowerRangeBound( void )
{
    return -100.0;
}

double sphereFunctionUpperRangeBound( void )
{
    return 100.0;
}

double ciasFunctionLowerRangeBound( void )
{
  return 0.0;
}

double ciasFunctionUpperRangeBound( void )
{
  return 1.0;
}