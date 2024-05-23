//
// Created by Renzo on 12-6-2020.
//

#include "shared/merge_sort.h"

/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *mergeSort( double *array, int array_size )
{
  int i, *sorted, *tosort;
  
  sorted = (int *) Malloc( array_size * sizeof( int ) );
  tosort = (int *) Malloc( array_size * sizeof( int ) );
  for( i = 0; i < array_size; i++ )
    tosort[i] = i;
  
  if( array_size == 1 )
    sorted[0] = 0;
  else
    mergeSortWithinBounds( array, sorted, tosort, 0, array_size-1 );
  
  free( tosort );
  
  return( sorted );
}

/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q )
{
  int r;
  
  if( p < q )
  {
    r = (p + q) / 2;
    mergeSortWithinBounds( array, sorted, tosort, p, r );
    mergeSortWithinBounds( array, sorted, tosort, r+1, q );
    mergeSortMerge( array, sorted, tosort, p, r+1, q );
  }
}

/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q )
{
  int i, j, k, first;
  
  i = p;
  j = r;
  for( k = p; k <= q; k++ )
  {
    first = 0;
    if( j <= q )
    {
      if( i < r )
      {
        if( array[tosort[i]] < array[tosort[j]] )
          first = 1;
      }
    }
    else
      first = 1;
    
    if( first )
    {
      sorted[k] = tosort[i];
      i++;
    }
    else
    {
      sorted[k] = tosort[j];
      j++;
    }
  }
  
  for( k = p; k <= q; k++ )
    tosort[k] = sorted[k];
}

/**
 * Sorts an array of objectives and constraints
 * using constraint domination and returns the
 * sort-order (small to large).
 */
int *mergeSortFitness( double *objectives, double *constraints, int number_of_solutions )
{
  int i, *sorted, *tosort;
  
  sorted = (int *) Malloc( number_of_solutions * sizeof( int ) );
  tosort = (int *) Malloc( number_of_solutions * sizeof( int ) );
  for( i = 0; i < number_of_solutions; i++ )
    tosort[i] = i;
  
  if( number_of_solutions == 1 )
    sorted[0] = 0;
  else
    mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, 0, number_of_solutions-1 );
  
  free( tosort );
  
  return( sorted );
}

/**
 * Subroutine of merge sort, sorts the part of the objectives and
 * constraints arrays between p and q.
 */
void mergeSortFitnessWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q )
{
  int r;
  
  if( p < q )
  {
    r = (p + q) / 2;
    mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, p, r );
    mergeSortFitnessWithinBounds( objectives, constraints, sorted, tosort, r+1, q );
    mergeSortFitnessMerge( objectives, constraints, sorted, tosort, p, r+1, q );
  }
}

/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortFitnessMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q )
{
  int i, j, k, first;
  
  i = p;
  j = r;
  for( k = p; k <= q; k++ )
  {
    first = 0;
    if( j <= q )
    {
      if( i < r )
      {
        if( betterFitness( objectives[tosort[i]], constraints[tosort[i]],
                           objectives[tosort[j]], constraints[tosort[j]] ) )
          first = 1;
      }
    }
    else
      first = 1;
    
    if( first )
    {
      sorted[k] = tosort[i];
      i++;
    }
    else
    {
      sorted[k] = tosort[j];
      j++;
    }
  }
  
  for( k = p; k <= q; k++ )
    tosort[k] = sorted[k];
}
