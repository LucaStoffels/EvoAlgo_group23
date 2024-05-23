//
// Created by Renzo on 12-6-2020.
//

#ifndef PROJECT_MATRIX_H
#define PROJECT_MATRIX_H

#include <math.h>
#include <stdio.h>

void *Malloc( long size );
double **matrixNew( int n, int m );
double vectorDotProduct( double *vector0, double *vector1, int n0 );
double *matrixVectorMultiplication( double **matrix, double *vector, int n0, int n1 );
double **matrixMatrixMultiplication( double **matrix0, double **matrix1, int n0, int n1, int n2 );
int blasDSWAP( int n, double *dx, int incx, double *dy, int incy );
int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy);
void blasDSCAL( int n, double sa, double x[], int incx );
int linpackDCHDC( double a[], int lda, int p, double work[], int ipvt[] );
double **choleskyDecomposition( double **matrix, int n );
int linpackDTRDI( double t[], int ldt, int n );
double **matrixLowerTriangularInverse( double **matrix, int n );
void matrixWriteToFile( FILE *file, double **matrix, int n0, int n1 );

#endif //PROJECT_MATRIX_H
