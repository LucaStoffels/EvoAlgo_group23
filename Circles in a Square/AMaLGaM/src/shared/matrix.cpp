//
// Created by Renzo on 12-6-2020.
//

#include "shared/matrix.h"

/**
 * Creates a new matrix with dimensions n x m.
 */
double **matrixNew( int n, int m )
{
  int      i;
  double **result;
  
  result = (double **) malloc( n*( sizeof( double * ) ) );
  for( i = 0; i < n; i++ )
    result[i] = (double *) malloc( m*( sizeof( double ) ) );
  
  return( result );
}

/**
 * Computes the dot product of two vectors of the same dimensionality n0.
 */
double vectorDotProduct( double *vector0, double *vector1, int n0 )
{
  int    i;
  double result;
  
  result = 0.0;
  for( i = 0; i < n0; i++ )
    result += vector0[i]*vector1[i];
  
  return( result );
}

/**
 * Computes the multiplication Av of a matrix A and a vector v
 * where matrix A has dimensions n0 x n1 and vector v has
 * dimensionality n1.
 */
double *matrixVectorMultiplication( double **matrix, double *vector, int n0, int n1 )
{
  int     i;
  double *result;
  
  result = (double *) malloc( n0*sizeof( double ) );
  for( i = 0; i < n0; i++ )
    result[i] = vectorDotProduct( matrix[i], vector, n1 );
  
  return( result );
}

/**
 * Computes the matrix multiplication of two matrices A and B
 * of dimensions A: n0 x n1 and B: n1 x n2.
 */
double **matrixMatrixMultiplication( double **matrix0, double **matrix1, int n0, int n1, int n2 )
{
  int     i, j, k;
  double **result;
  
  result = (double **) malloc( n0*sizeof( double * ) );
  for( i = 0; i < n0; i++ )
    result[i] = (double *) malloc( n2*sizeof( double ) );
  
  for( i = 0; i < n0; i++ )
  {
    for( j = 0; j < n2; j++ )
    {
      result[i][j] = 0;
      for( k = 0; k < n1; k++ )
        result[i][j] += matrix0[i][k]*matrix1[k][j];
    }
  }
  
  return( result );
}

/**
 * BLAS subroutine.
 */
int blasDSWAP( int n, double *dx, int incx, double *dy, int incy )
{
  double dtmp;
  
  if (n > 0)
  {
    incx *= sizeof( double );
    incy *= sizeof( double );
    
    dtmp  = (*dx);
    *dx   = (*dy);
    *dy   = dtmp;
    
    while( (--n) > 0 )
    {
      dx = (double *) ((char *) dx + incx);
      dy = (double *) ((char *) dy + incy);
      dtmp = (*dx); *dx = (*dy); *dy = dtmp;
    }
  }
  
  return( 0 );
}

/**
 * BLAS subroutine.
 */
int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy)
{
  double dtmp0, dtmp, *dx0, *dy0;
  
  if( n > 0 && da != 0. )
  {
    incx *= sizeof(double);
    incy *= sizeof(double);
    *dy  += da * (*dx);
    
    if( (n & 1) == 0 )
    {
      dx   = (double *) ((char *) dx + incx);
      dy   = (double *) ((char *) dy + incy);
      *dy += da * (*dx);
      --n;
    }
    n = n >> 1;
    while( n > 0 )
    {
      dy0   = (double *) ((char *) dy + incy);
      dy    = (double *) ((char *) dy0 + incy);
      dtmp0 = (*dy0);
      dtmp  = (*dy);
      dx0   = (double *) ((char *) dx + incx);
      dx    = (double *) ((char *) dx0 + incx);
      *dy0  = dtmp0 + da * (*dx0);
      *dy   = dtmp + da * (*dx);
      --n;
    }
  }
  
  return( 0 );
}

/**
 * BLAS subroutine.
 */
void blasDSCAL( int n, double sa, double x[], int incx )
{
  int i, ix, m;
  
  if( n <= 0 )
  {
  }
  else if( incx == 1 )
  {
    m = n % 5;
    
    for( i = 0; i < m; i++ )
    {
      x[i] = sa * x[i];
    }
    
    for( i = m; i < n; i = i + 5 )
    {
      x[i]   = sa * x[i];
      x[i+1] = sa * x[i+1];
      x[i+2] = sa * x[i+2];
      x[i+3] = sa * x[i+3];
      x[i+4] = sa * x[i+4];
    }
  }
  else
  {
    if( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }
    
    for( i = 0; i < n; i++ )
    {
      x[ix] = sa * x[ix];
      ix = ix + incx;
    }
  }
}

/**
 * LINPACK subroutine.
 */
int linpackDCHDC( double a[], int lda, int p, double work[], int ipvt[] )
{
  int    info, j, jp, k, l, maxl, pl, pu;
  double maxdia, temp;
  
  pl   = 1;
  pu   = 0;
  info = p;
  for( k = 1; k <= p; k++ )
  {
    maxdia = a[k-1+(k-1)*lda];
    maxl   = k;
    if( pl <= k && k < pu )
    {
      for( l = k+1; l <= pu; l++ )
      {
        if( maxdia < a[l-1+(l-1)*lda] )
        {
          maxdia = a[l-1+(l-1)*lda];
          maxl   = l;
        }
      }
    }
    
    if( maxdia <= 0.0 )
    {
      info = k - 1;
      
      return( info );
    }
    
    if( k != maxl )
    {
      blasDSWAP( k-1, a+0+(k-1)*lda, 1, a+0+(maxl-1)*lda, 1 );
      
      a[maxl-1+(maxl-1)*lda] = a[k-1+(k-1)*lda];
      a[k-1+(k-1)*lda]       = maxdia;
      jp                     = ipvt[maxl-1];
      ipvt[maxl-1]           = ipvt[k-1];
      ipvt[k-1]              = jp;
    }
    work[k-1]        = sqrt( a[k-1+(k-1)*lda] );
    a[k-1+(k-1)*lda] = work[k-1];
    
    for( j = k+1; j <= p; j++ )
    {
      if( k != maxl )
      {
        if( j < maxl )
        {
          temp                = a[k-1+(j-1)*lda];
          a[k-1+(j-1)*lda]    = a[j-1+(maxl-1)*lda];
          a[j-1+(maxl-1)*lda] = temp;
        }
        else if ( maxl < j )
        {
          temp                = a[k-1+(j-1)*lda];
          a[k-1+(j-1)*lda]    = a[maxl-1+(j-1)*lda];
          a[maxl-1+(j-1)*lda] = temp;
        }
      }
      a[k-1+(j-1)*lda] = a[k-1+(j-1)*lda] / work[k-1];
      work[j-1]        = a[k-1+(j-1)*lda];
      temp             = -a[k-1+(j-1)*lda];
      
      blasDAXPY( j-k, temp, work+k, 1, a+k+(j-1)*lda, 1 );
    }
  }
  
  return( info );
}

/**
 * Computes the lower-triangle Cholesky Decomposition
 * of a square, symmetric and positive-definite matrix.
 * Subroutines from LINPACK and BLAS are used.
 */
double **choleskyDecomposition( double **matrix, int n )
{
  int     i, j, k, info, *ipvt;
  double *a, *work, **result;
  
  a    = (double *) Malloc( n*n*sizeof( double ) );
  work = (double *) Malloc( n*sizeof( double ) );
  ipvt = (int *) Malloc( n*sizeof( int ) );
  
  k = 0;
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      a[k] = matrix[i][j];
      k++;
    }
    ipvt[i] = 0;
  }
  
  info = linpackDCHDC( a, n, n, work, ipvt );
  
  result = matrixNew( n, n );
  if( info != n ) /* Matrix is not positive definite */
  {
    k = 0;
    for( i = 0; i < n; i++ )
    {
      for( j = 0; j < n; j++ )
      {
        result[i][j] = i != j ? 0.0 : sqrt( matrix[i][j] );
        k++;
      }
    }
  }
  else
  {
    k = 0;
    for( i = 0; i < n; i++ )
    {
      for( j = 0; j < n; j++ )
      {
        result[i][j] = i < j ? 0.0 : a[k];
        k++;
      }
    }
  }
  
  free( ipvt );
  free( work );
  free( a );
  
  return( result );
}

/**
 * LINPACK subroutine.
 */
int linpackDTRDI( double t[], int ldt, int n )
{
  int    j, k, info;
  double temp;
  
  info = 0;
  for( k = n; 1 <= k; k-- )
  {
    if ( t[k-1+(k-1)*ldt] == 0.0 )
    {
      info = k;
      break;
    }
    
    t[k-1+(k-1)*ldt] = 1.0 / t[k-1+(k-1)*ldt];
    temp = -t[k-1+(k-1)*ldt];
    
    if ( k != n )
    {
      blasDSCAL( n-k, temp, t+k+(k-1)*ldt, 1 );
    }
    
    for( j = 1; j <= k-1; j++ )
    {
      temp = t[k-1+(j-1)*ldt];
      t[k-1+(j-1)*ldt] = 0.0;
      blasDAXPY( n-k+1, temp, t+k-1+(k-1)*ldt, 1, t+k-1+(j-1)*ldt, 1 );
    }
  }
  
  return( info );
}

/**
 * Computes the inverse of a matrix that is of
 * lower triangular form.
 */
double **matrixLowerTriangularInverse( double **matrix, int n )
{
  int     i, j, k, info;
  double *t, **result;
  
  t = (double *) Malloc( n*n*sizeof( double ) );
  
  k = 0;
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      t[k] = matrix[j][i];
      k++;
    }
  }
  
  info = linpackDTRDI( t, n, n );
  
  result = matrixNew( n, n );
  k = 0;
  for( i = 0; i < n; i++ )
  {
    for( j = 0; j < n; j++ )
    {
      result[j][i] = i > j ? 0.0 : t[k];
      k++;
    }
  }
  
  free( t );
  
  return( result );
}

/**
 * Writes the contents of a matrix of dimensions n0 x n1 to a file.
 */
void matrixWriteToFile( FILE *file, double **matrix, int n0, int n1 )
{
  int  i, j;
  char line_for_output[10000];
  
  sprintf( line_for_output, "[" );
  fputs( line_for_output, file );
  for( i = 0; i < n0; i++ )
  {
    sprintf( line_for_output, "[" );
    fputs( line_for_output, file );
    for( j = 0; j < n1; j++ )
    {
      sprintf( line_for_output, "%lf", matrix[i][j] );
      fputs( line_for_output, file );
      if( j < n1-1 )
      {
        sprintf( line_for_output, ", " );
        fputs( line_for_output, file );
      }
    }
    if( i == n0-1 )
      sprintf( line_for_output, "]" );
    else
      sprintf( line_for_output, "];" );
    fputs( line_for_output, file );
  }
  sprintf( line_for_output, "]\n" );
  fputs( line_for_output, file );
}