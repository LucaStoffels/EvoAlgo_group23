//
// Created by Renzo on 12-6-2020.
//

#include "shared/command_line.h"
#include "shared/problems.h"

/**
 * Parses and checks the command line.
 */
void interpretCommandLine( int argc, char **argv, short full_model, cmd_params & parameters )
{
  parseCommandLine( argc, argv, parameters );
  
  if( parameters.use_guidelines )
  {
    if(full_model){
      parameters.population_size = (int) (17.0 + 3.0*pow((double) parameters.number_of_parameters,1.5));
    } else {
      parameters.population_size = (int) (10.0*pow((double) parameters.number_of_parameters,0.5));
    }
    parameters.tau                              = 0.35;
    parameters.distribution_multiplier_decrease = 0.9;
    parameters.st_dev_ratio_threshold           = 1.0;
    parameters.maximum_no_improvement_stretch   = 25 + parameters.number_of_parameters;
  }
  
  checkOptions(parameters);
}

/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine( int argc, char **argv, cmd_params & parameters )
{
  int index;
  
  index = 1;
  
  parseOptions( argc, argv, &index, parameters );
  
  parseParameters( argc, argv, &index, parameters  );
}

/**
 * Parses only the options from the command line.
 */
void parseOptions( int argc, char **argv, int *index, cmd_params & parameters )
{
  double dummy;
  
  parameters.write_generational_statistics = 0;
  parameters.write_generational_solutions  = 0;
  parameters.print_verbose_overview        = 0;
  parameters.use_vtr                       = 0;
  parameters.use_guidelines                = 0;
  
  for( ; (*index) < argc; (*index)++ )
  {
    if( argv[*index][0] == '-' )
    {
      /* If it is a negative number, the option part is over */
      if( sscanf( argv[*index], "%lf", &dummy ) && argv[*index][1] != '\0' )
        break;
      
      if( argv[*index][1] == '\0' )
        optionError( argv, *index );
      else if( argv[*index][2] != '\0' )
        optionError( argv, *index );
      else
      {
        switch( argv[*index][1] )
        {
          case '?': printUsage(); break;
          case 'P': printProblems(); break;
          case 's': parameters.write_generational_statistics = 1; break;
          case 'w': parameters.write_generational_solutions  = 1; break;
          case 'v': parameters.print_verbose_overview        = 1; break;
          case 'r': parameters.use_vtr                       = 1; break;
          case 'g': parameters.use_guidelines                = 1; break;
          default : optionError( argv, *index );
        }
      }
    }
    else /* Argument is not an option, so option part is over */
      break;
  }
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError( char **argv, int index )
{
  printf("Illegal option: %s\n\n", argv[index]);
  
  printUsage();
}

/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters( int argc, char **argv, int *index, cmd_params & parameters )
{
  int noError;
  
  int params = (argc - *index);
  if( params != 13)
  {
    printf("Number of parameters is incorrect, require 13 parameters (you provided %d).\n\n", (argc - *index));
    
    printUsage();
  }
  
  noError = 1;
  noError = noError && sscanf( argv[*index+0], "%d", &(parameters.problem_index));
  noError = noError && sscanf( argv[*index+1], "%d",  &(parameters.number_of_parameters) );
  noError = noError && sscanf( argv[*index+2], "%lf", &(parameters.lower_user_range) );
  noError = noError && sscanf( argv[*index+3], "%lf", &(parameters.upper_user_range) );
  noError = noError && sscanf( argv[*index+4], "%lf", &(parameters.tau) );
  noError = noError && sscanf( argv[*index+5], "%d",  &(parameters.population_size) );
  noError = noError && sscanf( argv[*index+6], "%d",  &(parameters.number_of_populations) );
  noError = noError && sscanf( argv[*index+7], "%lf", &(parameters.distribution_multiplier_decrease) );
  noError = noError && sscanf( argv[*index+8], "%lf", &(parameters.st_dev_ratio_threshold) );
  noError = noError && sscanf( argv[*index+9], "%d",  &(parameters.maximum_number_of_evaluations) );
  noError = noError && sscanf( argv[*index+10], "%lf", &(parameters.vtr) );
  noError = noError && sscanf( argv[*index+11], "%d",  &(parameters.maximum_no_improvement_stretch) );
  noError = noError && sscanf( argv[*index+12], "%lf", &(parameters.fitness_variance_tolerance) );
  parameters.random_seed = time(0);
  
  if( !noError )
  {
    printf("Error parsing parameters.\n\n");
    
    printUsage();
  }
}

/**
 * Prints installed problems and exits the program.
 */
void printProblems( void )
{
    int i;

    printf("Installed problem index and name:\n");
    for ( i = 0; i < numberOfInstalledProblems(); i++ ) {
        printf("\t%d: %s\n", i, installedProblemName(i));
    }
    
    exit(0);
}

/**
 * Prints usage information and exits the program.
 */
void printUsage( void )
{
  printf("Usage: AMaLGaM-%s [-?] [-s] [-w] [-v] [-r] [-g] pro dim low upp tau pop nop dmd srt eva vtr imp tol\n", ALGORITHM_NAME);
  printf(" -?: Prints out this usage information.\n");
  printf(" -s: Enables computing and writing of statistics every generation.\n");
  printf(" -w: Enables writing of solutions and their fitnesses every generation.\n");
  printf(" -v: Enables verbose mode. Prints the settings before starting the run.\n");
  printf(" -r: Enables use of vtr in termination condition (value-to-reach).\n");
  printf(" -g: Uses guidelines to override parameter settings for those parameters\n");
  printf("     for which a guideline is known in literature. These parameters are:\n");
  printf("     tau pop dmd srt imp\n");
  printf("\n");
  printf("  pro: Problem index.\n");
  printf("  dim: Number of parameters.\n");
  printf("  low: Overall initialization lower bound.\n");
  printf("  upp: Overall initialization upper bound.\n");
  printf("  tau: Selection percentile (tau in [1/pop,1], truncation selection).\n");
  printf("  pop: Population size per normal.\n");
  printf("  nop: Number of populations.\n");
  printf("  dmd: The distribution multiplier decreaser (in (0,1), increaser is always 1/dmd).\n");
  printf("  srt: The standard-deviation ratio threshold for triggering variance-scaling.\n");
  printf("  eva: Maximum number of evaluations allowed.\n");
  printf("  vtr: The value to reach. If the objective value of the best feasible solution reaches\n");
  printf("       this value, termination is enforced (if -r is specified).\n");
  printf("  imp: Maximum number of subsequent generations without an improvement while the\n");
  printf("       the distribution multiplier is <= 1.0.\n");
  printf("  tol: The tolerance level for fitness variance (i.e. minimum fitness variance)\n");
  exit( 0 );
}

/**
 * Checks whether the selected options are feasible.
 */
void checkOptions( cmd_params parameters )
{
  if( parameters.number_of_parameters < 1 )
  {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", parameters.number_of_parameters);
    printf("\n\n");
    
    exit( 0 );
  }
  
  if( ((int) (parameters.tau*parameters.population_size)) <= 0 || parameters.tau >= 1 )
  {
    printf("\n");
    printf("Error: tau not in range (read: %e). Require tau in [1/pop,1] (read: [%e,%e]).", parameters.tau, 1.0/((double) parameters.population_size), 1.0);
    printf("\n\n");
    
    exit( 0 );
  }
  
  if( parameters.population_size < 1 )
  {
    printf("\n");
    printf("Error: population size < 1 (read: %d). Require population size >= 1.", parameters.population_size);
    printf("\n\n");
    
    exit( 0 );
  }
  
  if( parameters.number_of_populations < 1 )
  {
    printf("\n");
    printf("Error: number of populations < 1 (read: %d). Require number of populations >= 1.", parameters.number_of_populations);
    printf("\n\n");
    
    exit( 0 );
  }
  
  if( parameters.maximum_number_of_evaluations < 1 )
  {
    printf("\n");
    printf("Error: maximum number of evaluations < 1 (read: %d). Require maximum number of evaluations >= 1.", parameters.maximum_number_of_evaluations);
    printf("\n\n");
    
    exit( 0 );
  }
}

/**
 * Prints the settings as read on the command line.
 */
void printVerboseOverview( cmd_params parameters, double *lower_init_ranges, double *upper_init_ranges, double *lower_range_bounds, double *upper_range_bounds)
{
  int i;
  
  printf("### Settings #################################################\n");
  printf("#\n");
  printf("# Statistics writing every generation: %s\n", parameters.write_generational_statistics ? "enabled" : "disabled");
  printf("# Population file writing            : %s\n", parameters.write_generational_solutions ? "enabled" : "disabled");
  printf("# Use of value-to-reach (vtr)        : %s\n", parameters.use_vtr ? "enabled" : "disabled");
  printf("#\n");
  printf("##############################################################\n");
  printf("#\n");
  printf("# Optimization problem    = Circles in a Square\n");
  printf("# Number of parameters    = %d\n", parameters.number_of_parameters);
  printf("# Initialization ranges   = ");
  for( i = 0; i < parameters.number_of_parameters; i++ )
  {
    printf("x_%d: [%e;%e]", i, lower_init_ranges[i], upper_init_ranges[i]);
    if( i < parameters.number_of_parameters-1 )
      printf("\n#                           ");
  }
  printf("\n");
  printf("# Boundary ranges         = ");
  for( i = 0; i < parameters.number_of_parameters; i++ )
  {
    printf("x_%d: [%e;%e]", i, lower_range_bounds[i], upper_range_bounds[i]);
    if( i < parameters.number_of_parameters-1 )
      printf("\n#                           ");
  }
  printf("\n");
  printf("# Tau                     = %e\n", parameters.tau);
  printf("# Population size/normal  = %d\n", parameters.population_size);
  printf("# Number of populations   = %d\n", parameters.number_of_populations);
  printf("# Dis. mult. decreaser    = %e\n", parameters.distribution_multiplier_decrease);
  printf("# St. dev. rat. threshold = %e\n", parameters.st_dev_ratio_threshold);
  printf("# Maximum numb. of eval.  = %d\n", parameters.maximum_number_of_evaluations);
  printf("# Value to reach (vtr)    = %e\n", parameters.vtr);
  printf("# Max. no-improv. stretch = %d\n", parameters.maximum_no_improvement_stretch);
  printf("# Fitness var. tolerance  = %e\n", parameters.fitness_variance_tolerance);
  printf("# Random seed             = %ld\n", parameters.random_seed);
  printf("#\n");
  printf("##############################################################\n");
}