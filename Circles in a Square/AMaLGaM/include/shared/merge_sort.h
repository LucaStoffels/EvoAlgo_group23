//
// Created by Renzo on 12-6-2020.
//

#ifndef PROJECT_MERGE_SORT_H
#define PROJECT_MERGE_SORT_H

#include <stdlib.h>
#include "fitness.h"
#include "elementary.h"

int *mergeSort( double *array, int array_size );
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
int *mergeSortFitness( double *objectives, double *constraints, int number_of_solutions );
void mergeSortFitnessWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q );
void mergeSortFitnessMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q );

#endif //PROJECT_MERGE_SORT_H
