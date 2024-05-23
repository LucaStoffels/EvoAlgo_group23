//
// Created by Renzo on 23-6-2020.
//

#include "shared/random.h"

short haveNextNextGaussian = 0;
double nextNextGaussian;
extern int64_t random_seed_changing;

/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double randomRealUniform01( void )
{
  int64_t n26, n27;
  double  result;
  
  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n26                  = (int64_t)(random_seed_changing >> (48 - 26));
  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n27                  = (int64_t)(random_seed_changing >> (48 - 27));
  result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));
  
  return( result );
}

/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int randomInt( int maximum )
{
  int result;
  
  result = (int) (((double) maximum)*randomRealUniform01());
  
  return( result );
}

/**
 * Returns a random double, distributed normally with mean 0 and variance 1.
 */
double random1DNormalUnit( void )
{
  double v1, v2, s, multiplier, value;
  
  if( haveNextNextGaussian )
  {
    haveNextNextGaussian = 0;
    
    return( nextNextGaussian );
  }
  else
  {
    do
    {
      v1 = 2 * (randomRealUniform01()) - 1;
      v2 = 2 * (randomRealUniform01()) - 1;
      s = v1 * v1 + v2 * v2;
    } while (s >= 1);
    
    value                = -2 * log(s)/s;
    multiplier           = value <= 0.0 ? 0.0 : sqrt( value );
    nextNextGaussian     = v2 * multiplier;
    haveNextNextGaussian = 1;
    
    return( v1 * multiplier );
  }
}

/**
 * Returns a random double, distributed normally with given mean and variance.
 */
double random1DNormalParameterized( double mean, double variance )
{
  double result;
  
  result = mean + sqrt( variance )*random1DNormalUnit();
  
  return( result );
}