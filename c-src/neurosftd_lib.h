#ifndef NEUROSFTD_LIB
#define NEUROSFTD_LIB
#include "Python.h"
#include "numpy/arrayobject.h"

typedef unsigned int uint;

typedef struct tagMat
{
	npy_double* data;
	uint 	m;
	uint 	n;
	uint 	size;
}Matrix;

typedef struct tagNLayer
{
	Matrix*		w;
	Matrix*		c;
	Matrix*		bias;

	Matrix*		a;
	Matrix*		x_hat;
	Matrix*		sigd_vec;
	Matrix*		sigdd_vec;
	
	Matrix*		psi;
	Matrix*		out;
	Matrix*		out_grad;
	Matrix*		input;
	Matrix*		in_grad;

	Matrix* 	deda;
	Matrix*		dedw;
	Matrix*		prev_dw;
	Matrix*		dedpsi;
	Matrix*		dbias;
	Matrix*		prev_dbias;
	Matrix*		dedinput;
	Matrix*		dedgradin;
	Matrix*		dedc;
	Matrix*		prev_dc;

	npy_double	mommentum;
	npy_double	beta;
	uint		type;

	npy_double (* sig_eval)(npy_double);
	npy_double (* sig_deval)(npy_double);
	npy_double (* sig_ddeval)(npy_double);
}NLayer;

#endif
