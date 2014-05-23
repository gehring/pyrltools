#ifndef NEUROSFTD_LIB
#define NEUROSFTD_LIB
typedef unsigned int uint;

typedef struct tagMat
{
	double* data;
	uint 	m;
	uint 	n;
	uint 	size;
}Matrix;

typedef struct tagNLayer
{
	Matrix*		w;
	Matrix*		bias;

	Matrix*		a;
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

	double		mommentum;

	double (* sig_eval)(double);
	double (* sig_deval)(double);
	double (* sig_ddeval)(double);
}NLayer;
#endif
