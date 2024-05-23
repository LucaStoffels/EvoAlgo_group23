//
// Created by Renzo on 12-6-2020.
//

#include "shared/elementary.h"

/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *Malloc( long size )
{
  void *result;
  
  result = (void *) malloc( size );
  if( !result )
  {
    printf("\n");
    printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
    printf("\n");
    
    exit( 0 );
  }
  
  return( result );
}