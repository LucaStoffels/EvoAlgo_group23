//
// Created by Renzo on 12-6-2020.
//

// Define default algorithm name in case compiler doesnt set one
#ifndef ALGORITHM_NAME
#define ALGORITHM_NAME "full"
#endif //ALGORITHM_NAME

#ifndef PROJECT_COMMAND_LINE_H
#define PROJECT_COMMAND_LINE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctime>

// @todo check struct packing in terms of bytes for optimizations
struct cmd_params {
    int problem_index;                        /* Problem index to optimize. */
    short write_generational_statistics;      /* Whether to compute and write statistics every generation (0 = no). */
    short write_generational_solutions;       /* Whether to write the population every generation (0 = no). */
    short print_verbose_overview;             /* Whether to print a overview of settings (0 = no). */
    short use_vtr;                            /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
    short use_guidelines;                     /* Whether to override parameters with guidelines (for those that exist). */
    int number_of_parameters;                 /* The number of parameters to be optimized. */
    int population_size;                      /* The size of each population. */
    int number_of_populations;                /* The size of each population. */
    int maximum_number_of_evaluations;        /* The maximum number of evaluations. */
    int maximum_no_improvement_stretch;       /* The maximum number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
    double tau;                               /* The selection truncation percentile (in [1/population_size,1]). */
    double distribution_multiplier_decrease;  /* The multiplicative distribution multiplier decrease. */
    double st_dev_ratio_threshold;            /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
    double lower_user_range;                  /* The initial lower range-bound indicated by the user (same for all dimensions). */
    double upper_user_range;                  /* The initial upper range-bound indicated by the user (same for all dimensions). */
    double vtr;                               /* The value-to-reach (function value of best solution that is feasible). */
    double fitness_variance_tolerance;        /* The minimum fitness variance level that is allowed. */
    int64_t random_seed;                      /* The seed used for the random-number generator. */
};

void interpretCommandLine( int argc, char **argv, short full_model, cmd_params & parameters );
void parseCommandLine( int argc, char **argv, cmd_params & parameters  );
void parseOptions( int argc, char **argv, int *index, cmd_params & parameters );
void optionError( char **argv, int index );
void parseParameters( int argc, char **argv, int *index, cmd_params & parameters );
void printUsage( void );
void printProblems( void );
void checkOptions( cmd_params parameters );
void printVerboseOverview( cmd_params parameters, double *lower_init_ranges, double *upper_init_ranges, double *lower_range_bounds, double *upper_range_bounds);

#endif //PROJECT_COMMAND_LINE_H
