/**
 * AMaLGaM-Full.c
 *
 * Copyright (c) 1998-2010 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * Adapted Maximum-Likelihood Gaussian Model
 * Iterated Density-Estimation Evolutionary Algorithm
 * with a amalgam_full covariance matrix
 *
 * In this implementation, minimization is assumed.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - J�rn Grahl
 *
 * This is the most up-to-date literature reference regarding this software:
 *
 * P.A.N. Bosman. On Empirical Memory Design, Faster Selection of Bayesian
 * Factorizations and Parameter-Free Gaussian EDAs. In G. Raidl, E. Alba,
 * J. Bacardit, C. Bates Congdon, H.-G. Beyer, M. Birattari, C. Blum,
 * P.A.N. Bosman, D. Corne, C. Cotta, M. Di Penta, B. Doerr, R. Drechsler,
 * M. Ebner, J. Grahl, T. Jansen, J. Knowles, T. Lenaerts, M. Middendorf,
 * J.F. Miller, M. O'Neill, R. Poli, G. Squillero, K. Stanley, T. St�tzle
 * and J. van Hemert, editors, Proceedings of the Genetic and Evolutionary
 * Computation Conference - GECCO-2009, pages 389-396, ACM Press, New York,
 * New York, 2009.
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
#include "shared/elementary.h"
#include "shared/fitness.h"
#include "shared/matrix.h"
#include "shared/merge_sort.h"
#include "shared/command_line.h"
#include "shared/problems.h"
#include "shared/random.h"

void initialize( cmd_params & parameters );
void initializeMemory( void );
void initializeRandomNumberGenerator( void );
void initializeParameterRangeBounds( void );
void initializeDistributionMultipliers( void );
void initializePopulationsAndFitnessValues( void );
void initializeObjectiveRotationMatrix( void );
void computeRanks( void );
void computeRanksForOnePopulation( int population_index );
double distanceInParameterSpace( double *solution_a, double *solution_b );
void writeGenerationalStatistics( void );
void writeGenerationalSolutions( short final );
void writeGenerationalSolutionsBest( short final );
short checkTerminationCondition( void );
short checkNumberOfEvaluationsTerminationCondition( void );
short checkVTRTerminationCondition( void );
void determineBestSolutionInCurrentPopulations( int *population_of_best, int *index_of_best );
void checkFitnessVarianceTermination( void );
short checkFitnessVarianceTerminationSinglePopulation( int population_index );
void checkDistributionMultiplierTerminationCondition( void );
void makeSelections( void );
void makeSelectionsForOnePopulation( int population_index );
void makeSelectionsForOnePopulationUsingDiversityOnRank0( int population_index );
void makePopulations( void );
void estimateParametersAllPopulations( void );
void estimateParameters( int population_index );
void estimateParametersML( int population_index );
void estimateMeanVectorML( int population_index );
void estimateCovarianceMatrixML( int population_index );
void copyBestSolutionsToPopulations( void );
void applyDistributionMultipliers( void );
void generateAndEvaluateNewSolutionsToFillPopulations( void );
void computeParametersForSampling( int population_index );
double *generateNewSolution( int population_index );
void adaptDistributionMultipliers( void );
short generationalImprovementForOnePopulation( int population_index, double *st_dev_ratio );
double getStDevRatio( int population_index, double *parameters );
void ezilaitini( void );
void ezilaitiniMemory( void );
void ezilaitiniDistributionMultipliers( void );
void ezilaitiniObjectiveRotationMatrix( void );
void run( cmd_params parameters );
int main( int argc, char **argv );
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
short      write_generational_statistics,    /* Whether to compute and write statistics every generation (0 = no). */
write_generational_solutions,     /* Whether to write the population every generation (0 = no). */
print_verbose_overview,           /* Whether to print a overview of settings (0 = no). */
use_vtr,                          /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
use_guidelines,                   /* Whether to override parameters with guidelines (for those that exist). */
*populations_terminated;           /* Which populations have terminated (array). */
int        number_of_parameters,             /* The number of parameters to be optimized. */
population_size,                  /* The size of each population. */
number_of_populations,            /* The number of parallel populations that initially partition the search space. */
selection_size,                   /* The size of the selection for each population. */
maximum_number_of_evaluations,    /* The maximum number of evaluations. */
number_of_evaluations,            /* The current number of times a function evaluation was performed. */
number_of_generations,            /* The current generation count. */
problem_index,                    /* The index of the optimization problem. */
*samples_drawn_from_normal,        /* The number of samples drawn from the i-th normal in the last generation. */
*out_of_bounds_draws,              /* The number of draws that resulted in an out-of-bounds sample. */
*no_improvement_stretch,           /* The number of subsequent generations without an improvement while the distribution multiplier is <= 1.0, for each population separately. */
maximum_no_improvement_stretch;   /* The maximum number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
double     tau,                              /* The selection truncation percentile (in [1/population_size,1]). */
alpha_AMS,                        /* The percentile of offspring to apply AMS (anticipated mean shift) to. */
delta_AMS,                        /* The adaptation length for AMS (anticipated mean shift). */
*** populations,                      /* The populations containing the solutions. */
** objective_values,                 /* Objective values for population members. */
** constraint_values,                /* Sum of all constraint violations for population members. */
** ranks,                            /* Ranks of population members. */
*** selections,                       /* Selected solutions, one for each population. */
** objective_values_selections,      /* Objective values of selected solutions. */
** constraint_values_selections,     /* Sum of all constraint violations of selected solutions. */
* lower_range_bounds,               /* The respected lower bounds on parameters. */
* upper_range_bounds,               /* The respected upper bounds on parameters. */
* lower_init_ranges,                /* The initialization range lower bound. */
* upper_init_ranges,                /* The initialization range upper bound */
lower_user_range,                 /* The initial lower range-bound indicated by the user (same for all dimensions). */
upper_user_range,                 /* The initial upper range-bound indicated by the user (same for all dimensions). */
rotation_angle,                   /* The angle of rotation to be applied to the problem. */
* distribution_multipliers,         /* Distribution multipliers (AVS mechanism), one for each population. */
distribution_multiplier_increase, /* The multiplicative distribution multiplier increase. */
distribution_multiplier_decrease, /* The multiplicative distribution multiplier decrease. */
st_dev_ratio_threshold,           /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
vtr,                              /* The value-to-reach (function value of best solution that is feasible). */
fitness_variance_tolerance,       /* The minimum fitness variance level that is allowed. */
** mean_vectors,                     /* The mean vectors, one for each population. */
** mean_vectors_previous,            /* The mean vectors of the previous generation, one for each population. */
*** covariance_matrices,              /* The covariance matrices to be used for sampling, one for each population. */
*** cholesky_factors_lower_triangle,  /* The unique lower triangular matrix of the Cholesky factorization. */
** rotation_matrix;                  /* The rotation matrix to be applied before evaluating. */
int64_t    random_seed,                      /* The seed used for the random-number generator. */
random_seed_changing;             /* Internally used variable for randomly setting a random seed. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/




/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#define PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Random Numbers -=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Performs initializations that are required before starting a run.
 */
void initialize( cmd_params & parameters )
{
  number_of_generations = 0;
  number_of_evaluations = 0;
  
  problem_index = parameters.problem_index;
  write_generational_statistics = parameters.write_generational_statistics;
  write_generational_solutions = parameters.write_generational_solutions;
  number_of_populations = parameters.number_of_populations;
  use_vtr = parameters.use_vtr;
  maximum_number_of_evaluations = parameters.maximum_number_of_evaluations;
  maximum_no_improvement_stretch = parameters.maximum_no_improvement_stretch;
  distribution_multiplier_decrease = parameters.distribution_multiplier_decrease;
  st_dev_ratio_threshold = parameters.st_dev_ratio_threshold;
  lower_user_range = parameters.lower_user_range;
  upper_user_range = parameters.upper_user_range;
  vtr = parameters.vtr;
  fitness_variance_tolerance = parameters.fitness_variance_tolerance;
  random_seed = parameters.random_seed;
  tau = parameters.tau;
  population_size = parameters.population_size;
  number_of_parameters = parameters.number_of_parameters;
  
  alpha_AMS = 0.5*tau*(((double) population_size)/((double) (population_size-1)));
  delta_AMS = 2.0;
  
  initializeMemory();
  
  initializeRandomNumberGenerator();
  
  initializeParameterRangeBounds();
  
  initializeDistributionMultipliers();
  
  initializeObjectiveRotationMatrix();
  
  initializePopulationsAndFitnessValues();
  
  computeRanks();
}

/**
 * Initializes the memory.
 */
void initializeMemory( void )
{
  int i, j;
  
  selection_size = (int) (tau*(population_size));
  
  populations                      = (double ***) Malloc( number_of_populations*sizeof( double ** ) );
  populations_terminated           = (short *) Malloc( number_of_populations*sizeof( short ) );
  no_improvement_stretch           = (int *) Malloc( number_of_populations*sizeof( int ) );
  objective_values                 = (double **) Malloc( number_of_populations*sizeof( double * ) );
  constraint_values                = (double **) Malloc( number_of_populations*sizeof( double * ) );
  ranks                            = (double **) Malloc( number_of_populations*sizeof( double * ) );
  selections                       = (double ***) Malloc( number_of_populations*sizeof( double ** ) );
  objective_values_selections      = (double **) Malloc( number_of_populations*sizeof( double * ) );
  constraint_values_selections     = (double **) Malloc( number_of_populations*sizeof( double * ) );
  mean_vectors                     = (double **) Malloc( number_of_populations*sizeof( double * ) );
  mean_vectors_previous            = (double **) Malloc( number_of_populations*sizeof( double * ) );
  covariance_matrices              = (double ***) Malloc( number_of_populations*sizeof( double ** ) );
  cholesky_factors_lower_triangle  = (double ***) Malloc( number_of_populations*sizeof( double ** ) );
  
  for( i = 0; i < number_of_populations; i++ )
  {
    populations[i] = (double **) Malloc( population_size*sizeof( double * ) );
    for( j = 0; j < population_size; j++ )
      populations[i][j] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    populations_terminated[i] = 0;
    
    no_improvement_stretch[i] = 0;
    
    objective_values[i] = (double *) Malloc( population_size*sizeof( double ) );
    
    constraint_values[i] = (double *) Malloc( population_size*sizeof( double ) );
    
    ranks[i] = (double *) Malloc( population_size*sizeof( double ) );
    
    selections[i] = (double **) Malloc( selection_size*sizeof( double * ) );
    for( j = 0; j < selection_size; j++ )
      selections[i][j] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    objective_values_selections[i] = (double *) Malloc( selection_size*sizeof( double ) );
    
    constraint_values_selections[i] = (double *) Malloc( selection_size*sizeof( double ) );
    
    mean_vectors[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    mean_vectors_previous[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    covariance_matrices[i] = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( j = 0; j < number_of_parameters; j++ )
      covariance_matrices[i][j] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    cholesky_factors_lower_triangle[i] = NULL;
  }
  
  lower_range_bounds = (double *) Malloc( number_of_parameters*sizeof( double ) );
  upper_range_bounds = (double *) Malloc( number_of_parameters*sizeof( double ) );
  lower_init_ranges  = (double *) Malloc( number_of_parameters*sizeof( double ) );
  upper_init_ranges  = (double *) Malloc( number_of_parameters*sizeof( double ) );
}

/**
 * Initializes the random number generator.
 */
void initializeRandomNumberGenerator( void )
{
  struct tm *timep;
  time_t t;
  
  while( random_seed_changing == 0 )
  {
    t                    = time( NULL );
    timep                = localtime( &t );
    random_seed_changing = ((60*(long) timep->tm_min)) + (60*60*(long) timep->tm_hour) + ((long) timep->tm_sec);
    random_seed_changing = (random_seed_changing/((int) (9.99*randomRealUniform01())+1))*(((int) (randomRealUniform01()*1000000.0))%10);
  }
  
  random_seed = random_seed_changing;
}

/**
 * Initializes the parameter range bounds.
 */
void initializeParameterRangeBounds( void )
{
  int i;
  
  for( i = 0; i < number_of_parameters; i++ )
  {
    lower_range_bounds[i] = installedProblemLowerRangeBound( problem_index );
    upper_range_bounds[i] = installedProblemUpperRangeBound( problem_index );
  }
  
  for( i = 0; i < number_of_parameters; i++ )
  {
    lower_init_ranges[i] = lower_user_range;
    if( lower_user_range < lower_range_bounds[i] )
      lower_init_ranges[i] = lower_range_bounds[i];
    
    upper_init_ranges[i] = upper_user_range;
    if( upper_user_range > upper_range_bounds[i] )
      upper_init_ranges[i] = upper_range_bounds[i];
  }
}

/**
 * Initializes the distribution multipliers.
 */
void initializeDistributionMultipliers( void )
{
  int i;
  
  distribution_multipliers = (double *) Malloc( number_of_populations*sizeof( double ) );
  for( i = 0; i < number_of_populations; i++ )
    distribution_multipliers[i] = 1.0;
  
  samples_drawn_from_normal = (int *) Malloc( number_of_populations*sizeof( int ) );
  out_of_bounds_draws       = (int *) Malloc( number_of_populations*sizeof( int ) );
  
  distribution_multiplier_increase = 1.0/distribution_multiplier_decrease;
}

/**
 * Initializes the populations and the fitness values.
 */
void initializePopulationsAndFitnessValues( void )
{
  int     i, j, k, o, q, *sorted, ssize, j_min, *temporary_population_sizes;
  double *distances, d, d_min, **solutions, *fitnesses, *constraints, **leader_vectors;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    for( j = 0; j < population_size; j++ )
    {
      for( k = 0; k < number_of_parameters; k++ )
        populations[i][j][k] = lower_init_ranges[k] + (upper_init_ranges[k] - lower_init_ranges[k])*randomRealUniform01();
      
      installedProblemEvaluation( problem_index, number_of_parameters, populations[i][j], &(objective_values[i][j]), &(constraint_values[i][j]), number_of_evaluations );
    }
  }
  
  /* Initialize means and redistribute solutions */
  if( number_of_populations > 1 )
  {
    ssize                      = number_of_populations*population_size;
    solutions                  = (double **) Malloc( ssize*sizeof( double * ) );
    fitnesses                  = (double *) Malloc( ssize*sizeof( double ) );
    constraints                = (double *) Malloc( ssize*sizeof( double ) );
    temporary_population_sizes = (int *) Malloc( ssize*sizeof( int ) );
    distances                  = (double *) Malloc( ssize*sizeof( double ) );
    leader_vectors             = (double **) Malloc( number_of_populations*sizeof( double * ) );
    
    for( i = 0; i < ssize; i++ )
      solutions[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    for( i = 0; i < number_of_populations; i++ )
      leader_vectors[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    
    q = 0;
    for( i = 0; i < number_of_populations; i++ )
    {
      for( j = 0; j < population_size; j++ )
      {
        for( k = 0; k < number_of_parameters; k++ )
          solutions[q][k] = populations[i][j][k];
        
        fitnesses[q]   = objective_values[i][j];
        constraints[q] = constraint_values[i][j];
        q++;
      }
    }
    o = randomInt( number_of_parameters );
    
    for( i = 0; i < ssize; i++ )
      distances[i] = solutions[i][o];
    
    sorted = mergeSort( distances, ssize );
    
    q = 0;
    for( i = 0; i < number_of_parameters; i++ )
      leader_vectors[q][i] = solutions[sorted[0]][i];
    
    for( i = 0; i < ssize; i++ )
      distances[i] = -distanceInParameterSpace( leader_vectors[q], solutions[i] );
    
    free( sorted );
    
    q++;
    while( q < number_of_populations )
    {
      sorted = mergeSort( distances, ssize );
      
      for( i = 0; i < number_of_parameters; i++ )
        leader_vectors[q][i] = solutions[sorted[0]][i];
      
      for( i = 0; i < ssize; i++ )
      {
        d = -distanceInParameterSpace( leader_vectors[q], solutions[i] );
        if( d > distances[i] )
          distances[i] = d;
      }
      
      free( sorted );
      
      q++;
    }
    
    for( i = 0; i < number_of_populations; i++ )
      temporary_population_sizes[i] = 0;
    
    for( i = 0; i < ssize; i++ )
    {
      j_min = -1;
      d_min = 0;
      for( j = 0; j < number_of_populations; j++ )
      {
        if( temporary_population_sizes[j] < population_size )
        {
          d = distanceInParameterSpace( solutions[i], leader_vectors[j] );
          if( (j_min == -1) || (d < d_min) )
          {
            j_min = j;
            d_min = d;
          }
        }
      }
      for( k = 0; k < number_of_parameters; k++ )
        populations[j_min][temporary_population_sizes[j_min]][k]  = solutions[i][k];
      objective_values[j_min][temporary_population_sizes[j_min]]  = fitnesses[i];
      constraint_values[j_min][temporary_population_sizes[j_min]] = constraints[i];
      temporary_population_sizes[j_min]++;
    }
    
    for( i = 0; i < number_of_populations; i++ )
      free( leader_vectors[i] );
    free( leader_vectors );
    free( distances );
    free( temporary_population_sizes );
    free( fitnesses );
    free( constraints );
    for( i = 0; i < ssize; i++ )
      free( solutions[i] );
    free( solutions );
  }
}

/**
 * Computes the rotation matrix to be applied to any solution
 * before evaluating it (i.e. turns the evaluation functions
 * into rotated evaluation functions).
 */
void initializeObjectiveRotationMatrix( void )
{
  int      i, j, index0, index1;
  double **matrix, **product, theta, cos_theta, sin_theta;
  
  if( rotation_angle == 0.0 )
    return;
  
  matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
  for( i = 0; i < number_of_parameters; i++ )
    matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
  
  rotation_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
  for( i = 0; i < number_of_parameters; i++ )
    rotation_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
  
  /* Initialize the rotation matrix to the identity matrix */
  for( i = 0; i < number_of_parameters; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
      rotation_matrix[i][j] = 0.0;
    rotation_matrix[i][i] = 1.0;
  }
  
  /* Construct all rotation matrices (quadratic number) and multiply */
  theta     = (rotation_angle/180.0)*PI;
  cos_theta = cos( theta );
  sin_theta = sin( theta );
  for( index0 = 0; index0 < number_of_parameters-1; index0++ )
  {
    for( index1 = index0+1; index1 < number_of_parameters; index1++ )
    {
      for( i = 0; i < number_of_parameters; i++ )
      {
        for( j = 0; j < number_of_parameters; j++ )
          matrix[i][j] = 0.0;
        matrix[i][i] = 1.0;
      }
      matrix[index0][index0] = cos_theta;
      matrix[index0][index1] = -sin_theta;
      matrix[index1][index0] = sin_theta;
      matrix[index1][index1] = cos_theta;
      
      product = matrixMatrixMultiplication( matrix, rotation_matrix, number_of_parameters, number_of_parameters, number_of_parameters );
      for( i = 0; i < number_of_parameters; i++ )
        for( j = 0; j < number_of_parameters; j++ )
          rotation_matrix[i][j] = product[i][j];
      
      for( i = 0; i < number_of_parameters; i++ )
        free( product[i] );
      free( product );
    }
  }
  
  for( i = 0; i < number_of_parameters; i++ )
    free( matrix[i] );
  free( matrix );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Ranking -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Computes the ranks of the solutions in all populations.
 */
void computeRanks( void )
{
  int i;
  
  for( i = 0; i < number_of_populations; i++ )
    if( !populations_terminated[i] )
      computeRanksForOnePopulation( i );
}

/**
 * Computes the ranks of the solutions in one population.
 */
void computeRanksForOnePopulation( int population_index )
{
  int i, *sorted, rank;
  
  sorted = mergeSortFitness( objective_values[population_index], constraint_values[population_index], population_size );
  
  rank                               = 0;
  ranks[population_index][sorted[0]] = rank;
  for( i = 1; i < population_size; i++ )
  {
    if( objective_values[population_index][sorted[i]] != objective_values[population_index][sorted[i-1]] )
      rank++;
    
    ranks[population_index][sorted[i]] = rank;
  }
  
  free( sorted );
}

/**
 * Computes the distance between two solutions a and b as
 * the Euclidean distance in parameter space.
 */
double distanceInParameterSpace( double *solution_a, double *solution_b )
{
  int    i;
  double value, result;
  
  result = 0.0;
  for( i = 0; i < number_of_parameters; i++ )
  {
    value   = solution_b[i] - solution_a[i];
    result += value*value;
  }
  result = sqrt( result );
  
  return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void writeGenerationalStatistics( void )
{
  int     i, j;
  char    string[1000];
  double  overall_objective_avg, overall_objective_var, overall_objective_best, overall_objective_worst,
      overall_constraint_avg, overall_constraint_var, overall_constraint_best, overall_constraint_worst,
      *population_objective_avg, *population_objective_var, *population_objective_best, *population_objective_worst,
      *population_constraint_avg, *population_constraint_var, *population_constraint_best, *population_constraint_worst;
  FILE   *file;
  
  /* First compute the statistics */
  population_objective_avg    = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_constraint_avg   = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_objective_var    = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_constraint_var   = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_objective_best   = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_constraint_best  = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_objective_worst  = (double *) Malloc( number_of_populations*sizeof( double ) );
  population_constraint_worst = (double *) Malloc( number_of_populations*sizeof( double ) );
  
  /* Overall */
  /* Average, best and worst */
  overall_objective_avg    = 0.0;
  overall_constraint_avg   = 0.0;
  overall_objective_best   = objective_values[0][0];
  overall_objective_worst  = objective_values[0][0];
  overall_constraint_best  = constraint_values[0][0];
  overall_constraint_worst = constraint_values[0][0];
  for( i = 0; i < number_of_populations; i++ )
  {
    for( j = 0; j < population_size; j++ )
    {
      overall_objective_avg += objective_values[i][j];
      overall_constraint_avg += constraint_values[i][j];
      if( betterFitness( objective_values[i][j], constraint_values[i][j], overall_objective_best, overall_constraint_best ) )
      {
        overall_objective_best  = objective_values[i][j];
        overall_constraint_best = constraint_values[i][j];
      }
      if( betterFitness( overall_objective_worst, overall_constraint_worst, objective_values[i][j], constraint_values[i][j] ) )
      {
        overall_objective_worst  = objective_values[i][j];
        overall_constraint_worst = constraint_values[i][j];
      }
    }
  }
  overall_objective_avg = overall_objective_avg / ((double) (number_of_populations*population_size));
  overall_constraint_avg = overall_constraint_avg / ((double) (number_of_populations*population_size));
  
  /* Variance */
  overall_objective_var    = 0.0;
  overall_constraint_var   = 0.0;
  for( i = 0; i < number_of_populations; i++ )
  {
    for( j = 0; j < population_size; j++ )
    {
      overall_objective_var += (objective_values[i][j] - overall_objective_avg)*(objective_values[i][j] - overall_objective_avg);
      overall_constraint_var += (constraint_values[i][j] - overall_constraint_avg)*(constraint_values[i][j] - overall_constraint_avg);
    }
  }
  overall_objective_var = overall_objective_var / ((double) (number_of_populations*population_size));
  overall_constraint_var = overall_constraint_var / ((double) (number_of_populations*population_size));
  
  if( overall_objective_var <= 0.0 )
    overall_objective_var = 0.0;
  if( overall_constraint_var <= 0.0 )
    overall_constraint_var = 0.0;
  
  /* Per population */
  for( i = 0; i < number_of_populations; i++ )
  {
    /* Average, best and worst */
    population_objective_avg[i]    = 0.0;
    population_constraint_avg[i]   = 0.0;
    population_objective_best[i]   = objective_values[i][0];
    population_constraint_best[i]  = constraint_values[i][0];
    population_objective_worst[i]  = objective_values[i][0];
    population_constraint_worst[i] = constraint_values[i][0];
    for( j = 0; j < population_size; j++ )
    {
      population_objective_avg[i]  += objective_values[i][j];
      population_constraint_avg[i] += constraint_values[i][j];
      if( betterFitness( objective_values[i][j], constraint_values[i][j], population_objective_best[i], population_constraint_best[i] ) )
      {
        population_objective_best[i] = objective_values[i][j];
        population_constraint_best[i] = constraint_values[i][j];
      }
      if( betterFitness( population_objective_worst[i], population_constraint_worst[i], objective_values[i][j], constraint_values[i][j] ) )
      {
        population_objective_worst[i] = objective_values[i][j];
        population_constraint_worst[i] = constraint_values[i][j];
      }
    }
    population_objective_avg[i]  = population_objective_avg[i] / ((double) population_size);
    population_constraint_avg[i] = population_constraint_avg[i] / ((double) population_size);
    
    /* Variance */
    population_objective_var[i]    = 0.0;
    population_constraint_var[i]   = 0.0;
    for( j = 0; j < population_size; j++ )
    {
      population_objective_var[i]  += (objective_values[i][j] - population_objective_avg[i])*(objective_values[i][j] - population_objective_avg[i]);
      population_constraint_var[i] += (constraint_values[i][j] - population_constraint_avg[i])*(constraint_values[i][j] - population_constraint_avg[i]);
    }
    population_objective_var[i]  = population_objective_var[i] / ((double) population_size);
    population_constraint_var[i] = population_constraint_var[i] / ((double) population_size);
    
    if( population_objective_var[i] <= 0.0 )
      population_objective_var[i] = 0.0;
    if( population_constraint_var[i] <= 0.0 )
      population_constraint_var[i] = 0.0;
  }
  
  /* Then write them */
  file = NULL;
  if( number_of_generations == 0 )
  {
    file = fopen( "statistics.dat", "w" );
    
    sprintf( string, "# Generation Evaluations  Average-obj. Variance-obj.     Best-obj.    Worst-obj.  Average-con. Variance-con.     Best-con.    Worst-con.   [ ");
    fputs( string, file );
    
    for( i = 0; i < number_of_populations; i++ )
    {
      sprintf( string, "Pop.index     Dis.mult.  Pop.avg.obj.  Pop.var.obj. Pop.best.obj. Pop.worst.obj.  Pop.avg.con.  Pop.var.con. Pop.best.con. Pop.worst.con." );
      fputs( string, file );
      if( i < number_of_populations-1 )
      {
        sprintf( string, " | " );
        fputs( string, file );
      }
    }
    sprintf( string, " ]\n" );
    fputs( string, file );
  }
  else
    file = fopen( "statistics.dat", "a" );
  
  sprintf( string, "  %10d %11d %13e %13e %13e %13e %13e %13e %13e %13e   [ ", number_of_generations, number_of_evaluations, overall_objective_avg, overall_objective_var, overall_objective_best, overall_objective_worst, overall_constraint_avg, overall_constraint_var, overall_constraint_best, overall_constraint_worst );
  fputs( string, file );
  
  for( i = 0; i < number_of_populations; i++ )
  {
    sprintf( string, "%9d %13e %13e %13e %13e  %13e %13e %13e %13e  %13e", i, distribution_multipliers[i], population_objective_avg[i], population_objective_var[i], population_objective_best[i], population_objective_worst[i], population_constraint_avg[i], population_constraint_var[i], population_constraint_best[i], population_constraint_worst[i] );
    fputs( string, file );
    if( i < number_of_populations-1 )
    {
      sprintf( string, " | " );
      fputs( string, file );
    }
  }
  sprintf( string, " ]\n" );
  fputs( string, file );
  
  fclose( file );
  
  free( population_objective_avg );
  free( population_constraint_avg );
  free( population_objective_var );
  free( population_constraint_var );
  free( population_objective_best );
  free( population_constraint_best );
  free( population_objective_worst );
  free( population_constraint_worst );
}

/**
 * Writes the solutions to various files. The filenames
 * contain the generation. If the flag final is set
 * (final != 0), the generation number in the filename
 * is replaced with the word "final".
 *
 * all_populations_generation_xxxxx.dat : all populations combined
 * population_xxxxx_generation_xxxxx.dat: the individual populations
 * selection_xxxxx_generation_xxxxx.dat : the individual selections
 */
void writeGenerationalSolutions( short final )
{
  int   i, j, k;
  char  string[1000];
  FILE *file_all, *file_population, *file_selection;
  
  if( final )
    sprintf( string, "all_populations_generation_final.dat" );
  else
    sprintf( string, "all_populations_generation_%05d.dat", number_of_generations );
  file_all = fopen( string, "w" );
  
  for( i = 0; i < number_of_populations; i++ )
  {
    if( final )
      sprintf( string, "population_%05d_generation_final.dat", i );
    else
      sprintf( string, "population_%05d_generation_%05d.dat", i, number_of_generations );
    file_population = fopen( string, "w" );
    
    if( number_of_generations > 0 && !final )
    {
      sprintf( string, "selection_%05d_generation_%05d.dat", i, number_of_generations-1 );
      file_selection = fopen( string, "w" );
    }
    
    /* Populations */
    for( j = 0; j < population_size; j++ )
    {
      for( k = 0; k < number_of_parameters; k++ )
      {
        sprintf( string, "%13e", populations[i][j][k] );
        fputs( string, file_all );
        fputs( string, file_population );
        if( k < number_of_parameters-1 )
        {
          sprintf( string, " " );
          fputs( string, file_all );
          fputs( string, file_population );
        }
      }
      sprintf( string, "     " );
      fputs( string, file_all );
      fputs( string, file_population );
      sprintf( string, "%13e %13e", objective_values[i][j], constraint_values[i][j] );
      fputs( string, file_all );
      fputs( string, file_population );
      sprintf( string, "\n" );
      fputs( string, file_all );
      fputs( string, file_population );
    }
    
    fclose( file_population );
    
    /* Selections */
    if( number_of_generations > 0 && !final )
    {
      for( j = 0; j < selection_size; j++ )
      {
        for( k = 0; k < number_of_parameters; k++ )
        {
          sprintf( string, "%13e", selections[i][j][k] );
          fputs( string, file_selection );
          if( k < number_of_parameters-1 )
          {
            sprintf( string, " " );
            fputs( string, file_selection );
          }
          sprintf( string, "     " );
          fputs( string, file_selection );
        }
        sprintf( string, "%13e %13e", objective_values_selections[i][j], constraint_values_selections[i][j] );
        fputs( string, file_selection );
        sprintf( string, "\n" );
        fputs( string, file_selection );
      }
      fclose( file_selection );
    }
  }
  
  fclose( file_all );
  
  writeGenerationalSolutionsBest( final );
}

/**
 * Writes the best solution (measured in the single
 * available objective) to a file named
 * best_generation_xxxxx.dat where xxxxx is the
 * generation number. If the flag final is set
 * (final != 0), the generation number in the filename
 * is replaced with the word "final".The output
 * file contains the solution values with the
 * dimensions separated by a single white space,
 * followed by five white spaces and then the
 * single objective value for that solution
 * and its sum of constraint violations.
 */
void writeGenerationalSolutionsBest( short final )
{
  int   i, population_index_best, individual_index_best;
  char  string[1000];
  FILE *file;
  
  /* First find the best of all */
  determineBestSolutionInCurrentPopulations( &population_index_best, &individual_index_best );
  
  /* Then output it */
  if (final) {
      sprintf(string, "best_final.dat");
  }
  else {
      sprintf(string, "best_generation_%05d.dat", number_of_generations);
  }
  file = fopen( string, "w" );
  
  for( i = 0; i < number_of_parameters; i++ )
  {
    sprintf( string, "%13e", populations[population_index_best][individual_index_best][i] );
    fputs( string, file );
    if( i < number_of_parameters-1 )
    {
      sprintf( string, " " );
      fputs( string, file );
    }
  }
  sprintf( string, "     " );
  fputs( string, file );
  sprintf( string, "%13e %13e", objective_values[population_index_best][individual_index_best], constraint_values[population_index_best][individual_index_best] );
  fputs( string, file );
  sprintf( string, "\n" );
  fputs( string, file );
  
  fclose( file );
  
  printf("%d %.16f\n", number_of_evaluations, objective_values[population_index_best][individual_index_best]);
}

void writeMeanVectors(){
  int   i, j, population_index_best, individual_index_best;
  char  string[1000];
  FILE *file, *cov_file;

  /* First find the best of all */
  determineBestSolutionInCurrentPopulations( &population_index_best, &individual_index_best );

  /* Then output it */
  sprintf( string, "mean_vector_%05d.dat", number_of_generations );
  file = fopen( string, "w" );
  sprintf( string, "covariance_%05d.dat", number_of_generations );
  cov_file = fopen( string, "w" );

  for( j = 0; j < number_of_populations; j++){
    for( i = 0; i < number_of_parameters; i++ )
    {
      sprintf( string, "%13e", mean_vectors[j][i] );
      fputs( string, file );
      if( i < number_of_parameters-1 )
      {
        sprintf( string, " " );
        fputs( string, file );
      }
      for(int k = 0; k < number_of_parameters; k++){
        sprintf( string, "%13e", covariance_matrices[j][i][k]);
        fputs( string, cov_file );
        if( k < number_of_parameters-1 )
        {
          sprintf( string, " " );
          fputs( string, cov_file );
        }
      }
      sprintf( string, "\n" );
      fputs( string, cov_file );
    }
    sprintf( string, "\n" );
    fputs( string, file );
    sprintf( string, "\n" );
    fputs( string, cov_file );
  }
  sprintf( string, "\n" );
  fputs( string, file );

  fclose( file );
  fclose( cov_file );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if termination should be enforced, 0 otherwise.
 */
short checkTerminationCondition( void )
{
  short allTrue;
  int   i;
  
  if( checkNumberOfEvaluationsTerminationCondition() ){
    //std::cout << "Reached max eval" << std::endl;
    return( 1 );
  }
  
  if( use_vtr )
  {
    if( checkVTRTerminationCondition() ){
      //std::cout << "Reached vtr" << std::endl;
      return( 1 );
    }
  }
  
  checkFitnessVarianceTermination();
  
  checkDistributionMultiplierTerminationCondition();
  
  allTrue = 1;
  for( i = 0; i < number_of_populations; i++ )
  {
    if( !populations_terminated[i] )
    {
      allTrue = 0;
      break;
    }
  }
  
  return( allTrue );
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
short checkNumberOfEvaluationsTerminationCondition( void )
{
  if( number_of_evaluations >= maximum_number_of_evaluations )
    return( 1 );
  
  return( 0 );
}

/**
 * Returns 1 if the value-to-reach has been reached (in any population).
 */
short checkVTRTerminationCondition( void )
{
  int population_of_best, index_of_best;
  
  determineBestSolutionInCurrentPopulations( &population_of_best, &index_of_best );
  
  if( constraint_values[population_of_best][index_of_best] == 0 && objective_values[population_of_best][index_of_best] >= vtr  )
    return( 1 );
  
  return( 0 );
}

/**
 * Determines which solution is the best of all solutions
 * in all current populations.
 */
void determineBestSolutionInCurrentPopulations( int *population_of_best, int *index_of_best )
{
  int i, j;
  
  (*population_of_best) = 0;
  (*index_of_best)      = 0;
  for( i = 0; i < number_of_populations; i++ )
  {
    for( j = 0; j < population_size; j++ )
    {
      if( betterFitness( objective_values[i][j], constraint_values[i][j],
                         objective_values[(*population_of_best)][(*index_of_best)], constraint_values[(*population_of_best)][(*index_of_best)] ) )
      {
        (*population_of_best) = i;
        (*index_of_best)      = j;
      }
    }
  }
}

/**
 * Checks whether the fitness variance in any population
 * has become too small (user-defined tolerance).
 */
void checkFitnessVarianceTermination( void )
{
  int i;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    if( !populations_terminated[i] )
      if( checkFitnessVarianceTerminationSinglePopulation( i ) ){
        populations_terminated[i] = 1;
      }
  }
}

/**
 * Returns 1 if the fitness variance in a specific population
 * has become too small (user-defined tolerance).
 */
short checkFitnessVarianceTerminationSinglePopulation( int population_index )
{
  int    i;
  double objective_avg, objective_var;
  
  objective_avg = 0.0;
  for( i = 0; i < population_size; i++ )
    objective_avg  += objective_values[population_index][i];
  objective_avg = objective_avg / ((double) population_size);
  
  objective_var = 0.0;
  for( i = 0; i < population_size; i++ )
    objective_var  += (objective_values[population_index][i]-objective_avg)*(objective_values[population_index][i]-objective_avg);
  objective_var = objective_var / ((double) population_size);
  
  if( objective_var <= 0.0 )
    objective_var = 0.0;
  
  if( objective_var <= fitness_variance_tolerance ){
    //std::cout << "Reached fitness variance termination: " << objective_var << std::endl;
    return( 1 );
  }
  
  return( 0 );
}

/**
 * Checks whether the distribution multiplier in any population
 * has become too small (1e-10).
 */
void checkDistributionMultiplierTerminationCondition( void )
{
  int i;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    if( !populations_terminated[i] )
      if( distribution_multipliers[i] < 1e-10 ){
        //std::cout << "Reached distribution multiplier termination" << std::endl;
        populations_terminated[i] = 1;
      }
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Selection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Makes a set of selected solutions for each population.
 */
void makeSelections( void )
{
  int i;
  
  for( i = 0; i < number_of_populations; i++ )
    if( !populations_terminated[i] )
      makeSelectionsForOnePopulation( i );
}

/**
 * Performs truncation selection on a single population.
 */
void makeSelectionsForOnePopulation( int population_index )
{
  int i, j, *sorted;
  
  sorted = mergeSort( ranks[population_index], population_size );
  
  if( ranks[population_index][sorted[selection_size-1]] == 0 )
    makeSelectionsForOnePopulationUsingDiversityOnRank0( population_index );
  else
  {
    for( i = 0; i < selection_size; i++ )
    {
      for( j = 0; j < number_of_parameters; j++ )
        selections[population_index][i][j] = populations[population_index][sorted[i]][j];
      
      objective_values_selections[population_index][i]  = objective_values[population_index][sorted[i]];
      constraint_values_selections[population_index][i] = constraint_values[population_index][sorted[i]];
    }
  }
  
  free( sorted );
}

/**
 * Performs selection from all solutions that have rank 0
 * based on diversity.
 */
void makeSelectionsForOnePopulationUsingDiversityOnRank0( int population_index )
{
  int     i, j, number_of_rank0_solutions, *preselection_indices,
      *selection_indices, index_of_farthest, number_selected_so_far;
  double *nn_distances, distance_of_farthest, value;
  
  number_of_rank0_solutions = 0;
  for( i = 0; i < population_size; i++ )
  {
    if( ranks[population_index][i] == 0 )
      number_of_rank0_solutions++;
  }
  
  preselection_indices = (int *) Malloc( number_of_rank0_solutions*sizeof( int ) );
  j                    = 0;
  for( i = 0; i < population_size; i++ )
  {
    if( ranks[population_index][i] == 0 )
    {
      preselection_indices[j] = i;
      j++;
    }
  }
  
  index_of_farthest    = 0;
  distance_of_farthest = objective_values[population_index][preselection_indices[0]];
  for( i = 1; i < number_of_rank0_solutions; i++ )
  {
    if( objective_values[population_index][preselection_indices[i]] > distance_of_farthest )
    {
      index_of_farthest    = i;
      distance_of_farthest = objective_values[population_index][preselection_indices[i]];
    }
  }
  
  number_selected_so_far                    = 0;
  selection_indices                         = (int *) Malloc( selection_size*sizeof( int ) );
  selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
  preselection_indices[index_of_farthest]   = preselection_indices[number_of_rank0_solutions-1];
  number_of_rank0_solutions--;
  number_selected_so_far++;
  
  nn_distances = (double *) Malloc( number_of_rank0_solutions*sizeof( double ) );
  for( i = 0; i < number_of_rank0_solutions; i++ )
    nn_distances[i] = distanceInParameterSpace( populations[population_index][preselection_indices[i]], populations[population_index][selection_indices[number_selected_so_far-1]] );
  
  while( number_selected_so_far < selection_size )
  {
    index_of_farthest    = 0;
    distance_of_farthest = nn_distances[0];
    for( i = 1; i < number_of_rank0_solutions; i++ )
    {
      if( nn_distances[i] > distance_of_farthest )
      {
        index_of_farthest    = i;
        distance_of_farthest = nn_distances[i];
      }
    }
    
    selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
    preselection_indices[index_of_farthest]   = preselection_indices[number_of_rank0_solutions-1];
    nn_distances[index_of_farthest]           = nn_distances[number_of_rank0_solutions-1];
    number_of_rank0_solutions--;
    number_selected_so_far++;
    
    for( i = 0; i < number_of_rank0_solutions; i++ )
    {
      value = distanceInParameterSpace( populations[population_index][preselection_indices[i]], populations[population_index][selection_indices[number_selected_so_far-1]] );
      if( value < nn_distances[i] )
        nn_distances[i] = value;
    }
  }
  
  for( i = 0; i < selection_size; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
      selections[population_index][i][j] = populations[population_index][selection_indices[i]][j];
    
    objective_values_selections[population_index][i]  = objective_values[population_index][selection_indices[i]];
    constraint_values_selections[population_index][i] = constraint_values[population_index][selection_indices[i]];
  }
  
  free( nn_distances );
  free( selection_indices );
  free( preselection_indices );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * First estimates the parameters of a normal distribution in the
 * parameter space from the selected sets of solutions (a separate
 * normal distribution for each population). Then copies the single
 * best selected solutions to their respective populations. Finally
 * fills up each population, after the variances have been scaled,
 * by drawing new samples from the normal distributions and applying
 * AMS to several of these new solutions. Then, the fitness ranks
 * are recomputed. Finally, the distribution multipliers are adapted
 * according to the SDR-AVS mechanism.
 */
void makePopulations( void )
{
  estimateParametersAllPopulations();
  
  copyBestSolutionsToPopulations();
  
  applyDistributionMultipliers();
  
  generateAndEvaluateNewSolutionsToFillPopulations();
  
  computeRanks();
  
  adaptDistributionMultipliers();
}

/**
 * Estimates the parameters of the multivariate normal
 * distribution for each population separately.
 */
void estimateParametersAllPopulations( void )
{
  int i;
  
  for( i = 0; i < number_of_populations; i++ )
    if( !populations_terminated[i] )
      estimateParameters( i );
}

/**
 * Estimates the paramaters of the multivariate
 * normal distribution for a specified population.
 */
void estimateParameters( int population_index )
{
  estimateParametersML( population_index );
}

/**
 * Estimates (with maximum likelihood) the
 * parameters of a multivariate normal distribution
 * for a specified population.
 */
void estimateParametersML( int population_index )
{
  estimateMeanVectorML( population_index );
  
  estimateCovarianceMatrixML( population_index );
}

/**
 * Computes the sample mean for a specified population.
 */
void estimateMeanVectorML( int population_index )
{
  int i, j;
  
  if( number_of_generations > 0 )
  {
    for( i = 0; i < number_of_parameters; i++ )
      mean_vectors_previous[population_index][i] = mean_vectors[population_index][i];
  }
  
  for( i = 0; i < number_of_parameters; i++ )
  {
    mean_vectors[population_index][i] = 0.0;
    
    for( j = 0; j < selection_size; j++ )
      mean_vectors[population_index][i] += selections[population_index][j][i];
    
    mean_vectors[population_index][i] /= (double) selection_size;
  }
  
  /* Change the focus of the search to the best solution */
  if( distribution_multipliers[population_index] < 1.0 )
    for( i = 0; i < number_of_parameters; i++ )
      mean_vectors[population_index][i] = selections[population_index][0][i];
}

/**
 * Computes the matrix of sample covariances for
 * a specified population.
 *
 * It is important that the pre-condition must be satisified:
 * estimateMeanVector was called first.
 */
void estimateCovarianceMatrixML( int population_index )
{
  int i;
  
  /* First do the maximum-likelihood estimate from data */
  for( i = 0; i < number_of_parameters; i++ )
  {
    for( int j = i; j < number_of_parameters; j++ )
    {
      covariance_matrices[population_index][i][j] = 0.0;
      
      for( int k = 0; k < selection_size; k++ )
        covariance_matrices[population_index][i][j] += (selections[population_index][k][i]-mean_vectors[population_index][i])*(selections[population_index][k][j]-mean_vectors[population_index][j]);
      
      covariance_matrices[population_index][i][j] /= (double) selection_size;
    }
  }

  for( i = 0; i < number_of_parameters; i++ )
    for( int j = 0; j < i; j++ )
      covariance_matrices[population_index][i][j] = covariance_matrices[population_index][j][i];
}

/**
 * Copies the single very best of the selected solutions
 * to their respective populations.
 */
void copyBestSolutionsToPopulations( void )
{
  int i, j, k;
  
  for( i = 0; i < number_of_populations; i++ )
  {
      if (!populations_terminated[i])
      {
          for (j = 0; j < selection_size; j++) {
              for (k = 0; k < number_of_parameters; k++)
                  populations[i][j][k] = selections[i][j][k];

              objective_values[i][j] = objective_values_selections[i][j];
              constraint_values[i][j] = constraint_values_selections[i][j];
          }
      }
  }
}

/**
 * Applies the distribution multipliers.
 */
void applyDistributionMultipliers( void )
{
  int i, j, k;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    if( !populations_terminated[i] )
    {
      for( j = 0; j < number_of_parameters; j++ )
        for( k = 0; k < number_of_parameters; k++ )
          covariance_matrices[i][j][k] *= distribution_multipliers[i];
    }
  }
}

/**
 * Generates new solutions for each
 * of the populations in turn.
 */
void generateAndEvaluateNewSolutionsToFillPopulations( void )
{
  int     i;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    computeParametersForSampling( i );
    
    double* solution, *solution_AMS, shrink_factor;
    int number_of_AMS_solutions, j, k, q;
    short   out_of_range;

    solution_AMS = (double*)Malloc(number_of_parameters * sizeof(double));

    if( !populations_terminated[i] )
    {
      number_of_AMS_solutions      = (int) (alpha_AMS*(population_size-1));
      samples_drawn_from_normal[i] = 0;
      out_of_bounds_draws[i]       = 0;
      q                            = 0;
      for( j = selection_size; j < population_size; j++ )
      {
        solution = generateNewSolution( i );
        
        for( k = 0; k < number_of_parameters; k++ )
          populations[i][j][k] = solution[k];
        
        if( (number_of_generations > 0) && (q < number_of_AMS_solutions) )
        {
          out_of_range  = 1;
          shrink_factor = 2;
          while( (out_of_range == 1) && (shrink_factor > 1e-10) )
          {
            shrink_factor *= 0.5;
            out_of_range   = 0;
            for( k = 0; k < number_of_parameters; k++ )
            {
              solution_AMS[k] = solution[k] + shrink_factor*delta_AMS*distribution_multipliers[i]*(mean_vectors[i][k]-mean_vectors_previous[i][k]);

              if( !isParameterInRangeBounds(problem_index, solution_AMS[k] ) )
              {
                out_of_range = 1;
                break;
              }
            }
          }
          if( !out_of_range )
          {
            for( k = 0; k < number_of_parameters; k++ )
              populations[i][j][k] = solution_AMS[k];
          }
        }
        
        installedProblemEvaluation( problem_index, number_of_parameters, populations[i][j], &(objective_values[i][j]), &(constraint_values[i][j]), number_of_evaluations );
        
        q++;
        
        free( solution );
      }
    }

    free(solution_AMS);
  }
}

/**
 * Computes the Cholesky decompositions required for sampling
 * the multivariate normal distribution.
 */
void computeParametersForSampling( int population_index )
{
  int i;
  
  if( cholesky_factors_lower_triangle[population_index] )
  {
    for( i = 0; i < number_of_parameters; i++ )
      free( cholesky_factors_lower_triangle[population_index][i] );
    free( cholesky_factors_lower_triangle[population_index] );
  }
  
  cholesky_factors_lower_triangle[population_index] = choleskyDecomposition( covariance_matrices[population_index], number_of_parameters );
}

/**
 * Generates and returns a single new solution by drawing
 * a single sample from a specified model.
 */

//@todo aanpassen naar bounding box (>1 gaat naar 1)
//@todo check anticipated mean shift?!
double * generateNewSolution( int population_index )
{
  short   ready;
  int     i, times_not_in_bounds;
  double *result, *z;
  
  times_not_in_bounds = -1;
  out_of_bounds_draws[population_index]--;
  
  ready = 0;
  do
  {
    times_not_in_bounds++;
    samples_drawn_from_normal[population_index]++;
    out_of_bounds_draws[population_index]++;
    ready = 1;
    if( times_not_in_bounds >= 100 )
    {
      result = (double *) Malloc( number_of_parameters*sizeof( double ) );
      for( i = 0; i < number_of_parameters; i++ )
        result[i] = lower_init_ranges[i] + (upper_init_ranges[i] - lower_init_ranges[i])*randomRealUniform01();
    }
    else
    {
      z = (double *) Malloc( number_of_parameters*sizeof( double ) );
      
      for( i = 0; i < number_of_parameters; i++ )
        z[i] = random1DNormalUnit();
      
      result = matrixVectorMultiplication( cholesky_factors_lower_triangle[population_index], z, number_of_parameters, number_of_parameters );
      
      for( i = 0; i < number_of_parameters; i++ )
        result[i] += mean_vectors[population_index][i];
      
      free( z );

      // Check parameter range
      for (i = 0; i < number_of_parameters; i++)
      {
          if (!isParameterInRangeBounds( problem_index, result[i]))
              ready = 0;
      }
    }
  }
  while( !ready );
  
  return( result );
}

/**
 * Adapts the distribution multipliers according to
 * the SDR-AVS mechanism.
 */
void adaptDistributionMultipliers( void )
{
  short  improvement;
  int    i;
  double st_dev_ratio;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    if( !populations_terminated[i] )
    {
      if( (((double) out_of_bounds_draws[i])/((double) samples_drawn_from_normal[i])) > 0.9 )
        distribution_multipliers[i] *= 0.5;
      
      improvement = generationalImprovementForOnePopulation( i, &st_dev_ratio );
      
      if( improvement )
      {
        no_improvement_stretch[i] = 0;
        
        if( distribution_multipliers[i] < 1.0 )
          distribution_multipliers[i] = 1.0;
        
        if( st_dev_ratio > st_dev_ratio_threshold ){
          distribution_multipliers[i] *= distribution_multiplier_increase;
        }
      }
      else
      {
        if( distribution_multipliers[i] <= 1.0 )
          (no_improvement_stretch[i])++;
        
        if( (distribution_multipliers[i] > 1.0) || (no_improvement_stretch[i] >= maximum_no_improvement_stretch) )
          distribution_multipliers[i] *= distribution_multiplier_decrease;
        
        if( (no_improvement_stretch[i] < maximum_no_improvement_stretch) && (distribution_multipliers[i] < 1.0) )
          distribution_multipliers[i] = 1.0;
      }
    }
  }
}

/**
 * Determines whether an improvement is found for a specified
 * population. Returns 1 in case of an improvement, 0 otherwise.
 * The standard-deviation ratio required by the SDR-AVS
 * mechanism is computed and returned in the pointer variable.
 */
short generationalImprovementForOnePopulation( int population_index, double *st_dev_ratio )
{
  int     i, j, index_best_selected, index_best_population,
      number_of_improvements;
  double *average_parameters_of_improvements;
  
  /* Determine best selected solutions */
  index_best_selected = 0;
  for( i = 0; i < selection_size; i++ )
  {
    if( betterFitness( objective_values_selections[population_index][i], constraint_values_selections[population_index][i],
                       objective_values_selections[population_index][index_best_selected], constraint_values_selections[population_index][index_best_selected] ) )
      index_best_selected = i;
  }
  
  /* Determine best in the population and the average improvement parameters */
  average_parameters_of_improvements = (double *) Malloc( number_of_parameters*sizeof( double ) );
  for( i = 0; i < number_of_parameters; i++ )
    average_parameters_of_improvements[i] = 0.0;
  
  index_best_population   = 0;
  number_of_improvements  = 0;
  for( i = 0; i < population_size; i++ )
  {
    if( betterFitness( objective_values[population_index][i], constraint_values[population_index][i],
                       objective_values[population_index][index_best_population], constraint_values[population_index][index_best_population] ) )
      index_best_population = i;
    
    if( betterFitness( objective_values[population_index][i], constraint_values[population_index][i],
                       objective_values_selections[population_index][index_best_selected], constraint_values_selections[population_index][index_best_selected] ) )
    {
      number_of_improvements++;
      for( j = 0; j < number_of_parameters; j++ )
        average_parameters_of_improvements[j] += populations[population_index][i][j];
    }
  }
  
  /* Determine st.dev. ratio */
  *st_dev_ratio = 0.0;
  if( number_of_improvements > 0 )
  {
    for( i = 0; i < number_of_parameters; i++ )
      average_parameters_of_improvements[i] /= (double) number_of_improvements;
    
    *st_dev_ratio = getStDevRatio( population_index, average_parameters_of_improvements );
  }
  
  free( average_parameters_of_improvements );
  
  if( fabs( objective_values_selections[population_index][index_best_selected] - objective_values[population_index][index_best_population] ) == 0.0 )
    return( 0 );
  
  return( 1 );
}

/**
 * Computes and returns the standard-deviation-ratio
 * of a given point for a given model.
 */
double getStDevRatio( int population_index, double *parameters )
{
  int      i;
  double **inverse, result, *x_min_mu, *z;
  
  inverse = matrixLowerTriangularInverse( cholesky_factors_lower_triangle[population_index], number_of_parameters );
  
  x_min_mu = (double *) Malloc( number_of_parameters*sizeof( double ) );
  
  for( i = 0; i < number_of_parameters; i++ )
    x_min_mu[i] = parameters[i]-mean_vectors[population_index][i];
  
  z = matrixVectorMultiplication( inverse, x_min_mu, number_of_parameters, number_of_parameters );
  
  result = 0.0;
  for( i = 0; i < number_of_parameters; i++ )
  {
    if( fabs( z[i] ) > result )
      result = fabs( z[i] );
  }
  
  free( z );
  free( x_min_mu );
  for( i = 0; i < number_of_parameters; i++ )
    free( inverse[i] );
  free( inverse );
  
  return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitini( void )
{
  ezilaitiniMemory();
  
  ezilaitiniDistributionMultipliers();
  
  ezilaitiniObjectiveRotationMatrix();
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitiniMemory( void )
{
  int i, j;
  
  for( i = 0; i < number_of_populations; i++ )
  {
    for( j = 0; j < population_size; j++ )
      free( populations[i][j] );
    free( populations[i] );
    
    free( objective_values[i] );
    
    free( constraint_values[i] );
    
    free( ranks[i] );
    
    for( j = 0; j < selection_size; j++ )
      free( selections[i][j] );
    free( selections[i] );
    
    free( objective_values_selections[i] );
    
    free( constraint_values_selections[i] );
    
    free( mean_vectors[i] );
    
    free( mean_vectors_previous[i] );
    
    for( j = 0; j < number_of_parameters; j++ )
      free( covariance_matrices[i][j] );
    free( covariance_matrices[i] );
    
    if( cholesky_factors_lower_triangle[i] )
    {
      for( j = 0; j < number_of_parameters; j++ )
        free( cholesky_factors_lower_triangle[i][j] );
      free( cholesky_factors_lower_triangle[i] );
    }
  }
  
  free( covariance_matrices );
  free( cholesky_factors_lower_triangle );
  free( lower_range_bounds );
  free( upper_range_bounds );
  free( lower_init_ranges );
  free( upper_init_ranges );
  free( populations_terminated );
  free( no_improvement_stretch );
  free( populations );
  free( objective_values );
  free( constraint_values );
  free( ranks );
  free( selections );
  free( objective_values_selections );
  free( constraint_values_selections );
  free( mean_vectors );
  free( mean_vectors_previous );
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitiniDistributionMultipliers( void )
{
  free( distribution_multipliers );
  free( samples_drawn_from_normal );
  free( out_of_bounds_draws );
}

/**
 * Undoes initialization procedure by freeing up memory.
 */
void ezilaitiniObjectiveRotationMatrix( void )
{
  int i;
  
  if( rotation_angle == 0.0 )
    return;
  
  for( i = 0; i < number_of_parameters; i++ )
    free( rotation_matrix[i] );
  free( rotation_matrix );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Runs the IDEA.
 */
void run( cmd_params parameters )
{
  initialize(parameters);
  
  // Get the estimations and write to file
  makeSelections();
  estimateParametersAllPopulations();
  
  if( parameters.print_verbose_overview )
    printVerboseOverview(parameters, lower_init_ranges, upper_init_ranges, lower_range_bounds, upper_range_bounds);
  
  while( !checkTerminationCondition() )
  {
    if( write_generational_statistics )
      writeGenerationalStatistics();
    
    if( write_generational_solutions )
      writeGenerationalSolutions( 0 );
    
    makeSelections();
    
    makePopulations();
  
    if(write_generational_solutions)
      writeMeanVectors();
    
    number_of_generations++;
  }
  
  writeGenerationalStatistics();
  
  writeGenerationalSolutions( 1 );
  
  ezilaitini();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Main -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * The main function:
 * - interpret parameters on the command line
 * - run the algorithm with the interpreted parameters
 */
int main( int argc, char **argv )
{
  cmd_params parameters;
  
  interpretCommandLine( argc, argv, true, parameters );
  
  run(parameters);
  
  return( 0 );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/