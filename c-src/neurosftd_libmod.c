#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "neurosftd_lib.h"
#include <cblas.h>

typedef npy_double (*sig_fun)(npy_double);

inline uint id(uint i, uint j, uint m){
	return i*m + j;
}

PyArrayObject* matrix_toarray(Matrix* A){
	int nd = 2;
	npy_intp dims[2] = {A->n, A->m};
	PyArrayObject* arr = (PyArrayObject*) PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, A->data);
	if(arr == NULL){
		printf("%s\n", "Failed to generate np array from c matrix");
		return NULL;
	}
	return arr;
}

void array_tomatrix(PyObject* a, Matrix* A ){
	PyObject* arr = PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_INOUT_ARRAY);
	if(arr == NULL){
		printf("%s\n", "Failed to convert from np array to c matrix");
		return;
	}

	if( PyArray_NDIM(a) > 1){
		A->m = (uint) PyArray_DIM(a, 1);
	}else{
		A->m = 1;
	}
	A->n = (uint) PyArray_DIM(a, 0);
	A->size = A->n * A->m;
	A->data = (npy_double*) PyArray_DATA(arr);
	Py_DECREF(arr);

}

void matrix_mul(Matrix* A, Matrix* B, Matrix* C){
	cblas_dgemm(CblasRowMajor, 
				CblasNoTrans, 
				CblasNoTrans,
				A->n,
				B->m,
				A->m,
				1.0,
				A->data,
				A->m,
				B->data,
				B->m,
				0.0,
				C->data,
				C->m);
}


void matrix_trans_mul(Matrix* A, Matrix* B, Matrix* C){
	cblas_dgemm(CblasRowMajor, 
				CblasTrans, 
				CblasNoTrans,
				A->m,
				B->m,
				A->n,
				1.0,
				A->data,
				A->m,
				B->data,
				B->m,
				0.0,
				C->data,
				C->m);
}

void matrix_mul_trans(Matrix* A, Matrix* B, Matrix* C){
	cblas_dgemm(CblasRowMajor, 
				CblasNoTrans, 
				CblasTrans,
				A->n,
				B->n,
				A->m,
				1.0,
				A->data,
				A->m,
				B->data,
				B->m,
				0.0,
				C->data,
				C->m);
}

void matrix_vector_mul(Matrix* A, Matrix*B, Matrix* C, npy_double alpha, npy_double beta){
	cblas_dgemv(CblasRowMajor, 
		CblasNoTrans,
		A->n,
		A->m,
		alpha,
		A->data,
		A->m,
		B->data,
		1,
		beta,
		C->data,
		1);
}


npy_double col_dot(Matrix* A, uint i, Matrix* B, uint j){
	assert(A->n == B->n);
	
	npy_double sum = 0.0;
	uint k;
	for(k = 0; k<A->n; ++k){
		sum += A->data[id(k,i, A->m)]* B->data[id(k,j, B->m)];
	}
	return sum;
}

void map_array(sig_fun f, npy_double* in, npy_double* out, uint n){
	uint i;
	for( i =0; i<n; ++i){
		out[i] = (*f)(in[i]);
	}
}

void compute_gradient( NLayer* layer, Matrix* errors_sig, Matrix* errors_grad){
	npy_double* dsigmoids = layer->sigd_vec->data;
	npy_double ddsigmoids[layer->a->size];

	// map_array(layer->sig_deval, layer->a->data, dsigmoids, layer->a->size);
	map_array(layer->sig_ddeval, layer->a->data, ddsigmoids, layer->a->size);

	uint i, j;
	for( i = 0; i<layer->a->size; i++){
		// printf("%f, %f, %f\n", dsigmoids[i], ddsigmoids[i], layer->a->data[i]);
		layer->deda->data[i] =  ddsigmoids[i] * col_dot(layer->psi, i, errors_grad, i) 
									+ dsigmoids[i] * errors_sig->data[i];
		layer->dbias->data[i] = layer->deda->data[i];				
	}

	assert(layer->dedpsi->m == layer->a->size);
	for(i = 0; i<layer->dedpsi->n; ++i){
		for( j = 0; j<layer->dedpsi->m; ++j){
			layer->dedpsi->data[id(i, j, layer->dedpsi->m)] = errors_grad->data[id(i,j, errors_grad->m)]
																* dsigmoids[j];
		}
	}
	// printf("%s\n", "5");
	matrix_trans_mul(layer->dedpsi, layer->in_grad, layer->dedw);

	assert(layer->dedw->m == layer->input->size);
	assert(layer->dedw->n == layer->a->size);
	for( i = 0; i<layer->dedw->n; ++i){
		for( j =0; j<layer->dedw->m; ++j){
			layer->dedw->data[id(i,j,layer->dedw->m)] += layer->input->data[j] 
																* layer->deda->data[i];
		}
	}
	// printf("%s\n", "6");
	matrix_trans_mul(layer->w, layer->deda, layer->dedinput);
	// printf("%s\n", "7");
	matrix_mul(layer->dedpsi, layer->w, layer->dedgradin);
	// printf("%s\n", "8");
}

void compute_gradient_quadratic(NLayer* layer, Matrix* errors_sig, Matrix* errors_grad){
	npy_double* dsigmoids = layer->sigd_vec->data;
	npy_double ddsigmoids[layer->a->size];
	map_array(layer->sig_ddeval, layer->a->data, ddsigmoids, layer->a->size);

	uint i, j, k;
	// compute de/da and set de/dbias to de/da
	for( i = 0; i<layer->a->size; i++){
		// printf("%f, %f, %f\n", dsigmoids[i], ddsigmoids[i], layer->a->data[i]);
		layer->deda->data[i] =  ddsigmoids[i] * col_dot(layer->psi, i, errors_grad, i) 
									+ dsigmoids[i] * errors_sig->data[i];
		layer->dbias->data[i] = layer->deda->data[i];				
	}

	// compute de/dpsi 
	for(i = 0; i<layer->dedpsi->n; ++i){
		for( j = 0; j<layer->dedpsi->m; ++j){
			layer->dedpsi->data[id(i, j, layer->dedpsi->m)] = errors_grad->data[id(i,j, errors_grad->m)]
																* dsigmoids[j];
		}
	}

	// compute de/dw and initialize de/dc
	matrix_trans_mul(layer->dedpsi, layer->in_grad, layer->dedw);
	matrix_trans_mul(layer->dedpsi, layer->in_grad, layer->dedc);
	for( i = 0; i<layer->dedw->n; ++i){
		for( j =0; j<layer->dedw->m; ++j){
			uint index = id(i,j,layer->dedw->m);
			npy_double x_hat = layer->x_hat->data[index];
			layer->dedw->data[index] = x_hat * x_hat * layer->deda->data[i] +
										layer->dedw->data[index] * x_hat;
		}
	}
	// compute de/dc
	for( i = 0; i<layer->dedc->n; ++i){
		for( j =0; j<layer->dedc->m; ++j){
			uint index = id(i,j,layer->dedc->m);
			npy_double x_hat = layer->x_hat->data[index];
			npy_double w = layer->w->data[index];
			layer->dedw->data[index] = - 2 * x_hat * layer->deda->data[i] * w -
										layer->dedw->data[index] * w;
		}
	}

	// compute de/dsig for next layer
	for( i = 0; i<layer->dedinput->size; ++i){
		layer->dedinput->data[i] = 0.0;
		for( j = 0; j<layer->deda->size; ++j){
			uint index = id(j,i, layer->w->m);
			layer->dedinput->data[i] += 2* layer->deda->data[i] * layer->w->data[index]*
											layer->x_hat->data[index];
		}
	}

	// compute de/dgrad_sig for next layer
	for( i=0; i<layer->dedgradin->size; ++i){
		layer->dedgradin->data[i] = 0.0;
	}
	for( i = 0; i<layer->dedinput->size; ++i){
		for( j = 0; j<layer->deda->size; ++j){
			uint index = id(j,i, layer->w->m);
			for( k=0; k<layer->dedgradin->n; ++k){
				layer->dedgradin->data[id(k,i, layer->dedgradin->m)] += layer->dedpsi->data[id(k,i, layer->dedpsi->m)]*
																			layer->w->data[index] * layer->x_hat->data[index];
			}
		}
	}



}

void compute_gradient_from_np(PyObject* self, PyObject* args){
	PyObject* layer_cap;
	PyObject* errors_grad;
	PyObject* errors_sig;

	if(!PyArg_ParseTuple(args, "OOO", &layer_cap, &errors_sig, &errors_grad)){
		printf("%s\n", "Impossible parsing of argument in compute_gradient_from_np");
		return;
	} 
	
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	Matrix errors_grad_tmp, errors_sig_tmp;
	array_tomatrix( errors_grad, &errors_grad_tmp);
	array_tomatrix( errors_sig, &errors_sig_tmp);

	switch(layer->type){
		case 0:
			compute_gradient(layer, &errors_sig_tmp, &errors_grad_tmp);
			break;
		case 1:
			compute_gradient_quadratic(layer, &errors_sig_tmp, &errors_grad_tmp);
			break;
	}

	return Py_BuildValue("");
}

void evaluate_layer(NLayer* layer){
	uint i, j;
	// printf("1\n");
	for(i =0; i<layer->a->size; ++i){
		layer->a->data[i] = layer->bias->data[i];
	}
	// printf("2\n");
	matrix_vector_mul(layer->w, layer->input, layer->a, 1.0, 1.0);
	// printf("3\n");
	map_array(layer->sig_eval, layer->a->data, layer->out->data, layer->a->size);
	// printf("4\n");
	map_array(layer->sig_deval, layer->a->data, layer->sigd_vec->data, layer->a->size);
	// printf("5\n");
	matrix_mul_trans(layer->in_grad, layer->w, layer->psi);
	// printf("6\n");

	for(i = 0; i<layer->psi->n; ++i){
		for(j = 0; j<layer->psi->m; ++j){
			layer->out_grad->data[id(i, j, layer->out_grad->m)] = layer->psi->data[id(i,j, layer->psi->m)]
																	* layer->sigd_vec->data[j];
		}
	}
	// printf("7\n");

}

void evaluate_quad_layer(NLayer* layer){
	uint i, j, k;

	// offset inputs for all nodes and store as x_hat
	for( i = 0; i<layer->x_hat->n; ++i){
		for( j = 0; j<layer->x_hat->m; ++j){
			layer->x_hat->data[id(i,j,layer->x_hat->m)] = (layer->input->data[j] -
													layer->c->data[id(i,j,layer->c->m)]);
		}
	}

	// // compute a
	// for(i =0; i<layer->a->size; ++i){
	// 	layer->a->data[i] = layer->bias->data[i];
	// 	for(j = 0; j<layer->w->m; ++j){
	// 		npy_double x_hat = layer->x_hat->data[id(i,j,layer->x_hat->m)];
	// 		layer->a->data[i] += layer->w->data[id(i,j,layer->w->m)] * x_hat * x_hat;
	// 	}
	// }
	// // not cache friendly, needs to be reordered ordering...
	// // compute psi
	// for( i=0; i<layer->a->size; ++i){
	// 	for(k=0; k<layer->psi->n; ++k){
	// 			layer->psi->data[id(k,i, layer->psi->m)] = 0.0;
	// 	}
	// 	for(j=0; j<layer->w->m; ++j){
	// 		npy_double w = layer->w->data[id(i,j,layer->w->m)] * layer->x_hat->data[id(i,j,layer->x_hat->m)];
	// 		for(k=0; k<layer->psi->n; ++k){
	// 			layer->psi->data[id(k,i, layer->psi->m)] += w * layer->in_grad->data[id(k,j, layer->in_grad->m)];
	// 		}
	// 	}
	// }

	// // compute dsig and sig
	// map_array(layer->sig_eval, layer->a->data, layer->out->data, layer->a->size);
	// map_array(layer->sig_deval, layer->a->data, layer->sigd_vec->data, layer->a->size);

	// // compute out gradient
	// for(i = 0; i< layer->psi->n; ++i){
	// 	for(j=0; j<layer->psi->m; ++j){
	// 		uint index = id(i,j,layer->psi->m);
	// 		layer->out_grad->data[index] = layer->sigd_vec->data[j] * layer->psi->data[index];
	// 	}
	// }

}

void evaluate_layer_from_np(PyObject* self, PyObject* args){
	PyObject* layer_cap, *input, *in_grad;
	if(!PyArg_ParseTuple(args, "OOO", &layer_cap, &input, &in_grad)){
		printf("%s\n", "Impossible parsing of argument in evaluate_layer_from_np");
		return;
	} 
	
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	Matrix input_tmp, in_grad_tmp;
	array_tomatrix( in_grad, &in_grad_tmp);
	array_tomatrix( input, &input_tmp);

	uint i;
	for(i = 0; i<input_tmp.size; ++i){
		layer->input->data[i] = input_tmp.data[i];
	}

	for(i = 0; i<in_grad_tmp.size; ++i){
		layer->in_grad->data[i] = in_grad_tmp.data[i];
	}
	switch(layer->type){
		case 0:
			evaluate_layer(layer);
			break;
		case 1:
			evaluate_quad_layer(layer);
			break;
	}
	

	return Py_BuildValue("");
}

void update_weights_wmommentum(NLayer* layer, npy_double alpha, npy_double mom){
	uint i;
	for ( i = 0; i < layer->prev_dw->size; ++i){
		layer->prev_dw->data[i] = layer->prev_dw->data[i]*mom - layer->dedw->data[i] * alpha;
	}
	for ( i = 0; i < layer->prev_dbias->size; ++i){
		layer->prev_dbias->data[i] = layer->prev_dbias->data[i]*mom - layer->dbias->data[i] * alpha;
	}
	for ( i = 0; i < layer->prev_dw->size; ++i){
		layer->w->data[i] += layer->prev_dw->data[i];
	}
	for ( i = 0; i < layer->prev_dbias->size; ++i){
		layer->bias->data[i] += layer->prev_dbias->data[i];
	}
}

void update_quad_weights_wmommentum(NLayer* layer, npy_double alpha, npy_double mom){
	update_weights_wmommentum(layer, alpha, mom);
	uint i;
	for ( i = 0; i < layer->prev_dc->size; ++i){
		layer->prev_dc->data[i] = layer->prev_dc->data[i]*mom - layer->dedc->data[i] * alpha;
	}
	for ( i = 0; i < layer->prev_dc->size; ++i){
		layer->c->data[i] += layer->prev_dc->data[i];
	}
}

void update_weights_from_py(PyObject* self, PyObject* args){
	PyObject* layer_cap;
	npy_double alpha;
	if(!PyArg_ParseTuple(args, "Od", &layer_cap, &alpha)){
		printf("%s\n", "Impossible parsing of argument in update_weights_no_input");
		return;
	} 

	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	switch(layer->type){
		case 0:
			update_weights_wmommentum(layer, alpha, layer->mommentum);
			break;
		case 1:
			update_quad_weights_wmommentum(layer, alpha, layer->mommentum);
			break;
	}
	
	return Py_BuildValue("");
}

void set_mat_from_buffer(Matrix* A, npy_double* buf, uint offset){
	uint i;
	for(i = 0; i<A->size; ++i){
		A->data[i] = buf[i+offset];
	}
}

Matrix* create_matrix(uint n, uint m){
	Matrix* mat = malloc(sizeof(Matrix));
	mat->data = malloc(sizeof(npy_double) * n * m);
	mat->m = m;
	mat->n = n;
	mat->size = m*n;


	uint i;
	for(i = 0; i<mat->size; ++i){
		mat->data[i] = 0;
	}


	return mat;
}

void destroy_matrix(Matrix* A){
	free(A->data);
	free(A);
}

PyObject* create_layer(PyObject* self, PyObject* args){

	uint input_size;
	uint layer_input; 
	uint num_neuron;
	PyObject* init_weights, *w_hold;
	PyObject* init_bias, *b_hold; 
	npy_double mommentum;
	PyObject* sig_cap;
	PyObject* sigd_cap;
	PyObject* sigdd_cap;
	uint type;

	if(!PyArg_ParseTuple(args, "IIIOOdOOOI", &input_size, 
											&layer_input, 
											&num_neuron,
											&w_hold,
											&b_hold, 
											&mommentum,
											&sig_cap,
											&sigd_cap,
											&sigdd_cap,
											&type)){
		return NULL;
	}
	init_weights = PyArray_FROM_OTF(w_hold, NPY_DOUBLE, NPY_IN_ARRAY);
	init_bias = PyArray_FROM_OTF(b_hold, NPY_DOUBLE, NPY_IN_ARRAY);
	assert(init_weights != NULL);
	assert(init_bias != NULL);

	NLayer* layer = (NLayer*) malloc(sizeof(NLayer));
	layer->type = type;

	layer->a = create_matrix(num_neuron, 1);
	layer->sigd_vec = create_matrix(num_neuron, 1);
	layer->sigdd_vec = create_matrix(num_neuron, 1);
	layer->deda = create_matrix(num_neuron, 1);

	layer->w = create_matrix(num_neuron, layer_input);
	npy_double* tmp = (npy_double*) PyArray_DATA( init_weights);
	set_mat_from_buffer(layer->w, tmp, 0);

	if(type == 1){
		layer->c = create_matrix(num_neuron, layer_input);
		set_mat_from_buffer(layer->c, tmp, layer->w->size);
		layer->dedc = create_matrix(num_neuron, layer_input);
		layer->prev_dc = create_matrix(num_neuron, layer_input);
		layer->x_hat = create_matrix(num_neuron, layer_input);
	}

	layer->bias = create_matrix(num_neuron, 1);
	tmp = (npy_double*) PyArray_DATA(init_bias);
	set_mat_from_buffer(layer->bias, tmp, 0);

	layer->dedinput = create_matrix(layer_input, 1);
	layer->dedgradin = create_matrix(input_size, layer_input);
	
	layer->psi = create_matrix(input_size, num_neuron);
	layer->dedpsi = create_matrix(input_size, num_neuron);
	layer->out_grad = create_matrix(input_size, num_neuron);

	layer->out = create_matrix(num_neuron, 1);
	layer->input = create_matrix(layer_input, 1);
	layer->in_grad = create_matrix(input_size, layer_input);

	layer->sig_eval = (sig_fun) PyCapsule_GetPointer(sig_cap, "function_pointer");
	layer->sig_deval = (sig_fun) PyCapsule_GetPointer(sigd_cap, "function_pointer");
	layer->sig_ddeval = (sig_fun) PyCapsule_GetPointer(sigdd_cap, "function_pointer");

	layer->prev_dw = create_matrix(num_neuron, layer_input);
	layer->dedw = create_matrix(num_neuron, layer_input);
	layer->prev_dbias = create_matrix(num_neuron, 1);
	layer->dbias = create_matrix(num_neuron, 1);

	layer->mommentum = mommentum;

	Py_DECREF(init_weights);
	Py_DECREF(init_bias);
	
	return PyCapsule_New(layer, "NLayer", NULL);
}

void destroy_layer(PyObject* self, PyObject* args){
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		printf("%s\n", "Impossible parsing of argument in update_weights_no_input");
		return;
	} 

	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");

	destroy_matrix(layer->a);
	destroy_matrix(layer->deda);

	destroy_matrix(layer->sigd_vec);
	destroy_matrix(layer->sigdd_vec);

	destroy_matrix(layer->w);

	destroy_matrix(layer->bias);

	destroy_matrix(layer->dedinput);
	destroy_matrix(layer->dedgradin);

	destroy_matrix(layer->psi);
	destroy_matrix(layer->dedpsi);
	destroy_matrix(layer->out_grad);

	destroy_matrix(layer->out);
	destroy_matrix(layer->input);
	destroy_matrix(layer->in_grad);

	destroy_matrix(layer->prev_dw);
	destroy_matrix(layer->dedw);
	destroy_matrix(layer->prev_dbias);
	destroy_matrix(layer->dbias);

	if (layer->type == 1){
		destroy_matrix(layer->c);
		destroy_matrix(layer->dedc);
		destroy_matrix(layer->prev_dc);
		destroy_matrix(layer->x_hat);
	}

	// PyObject* seq = PySequence_Fast(list, "expected a sequence");
	// int len = PySequence_Size(list);
	// int i;
	// for( i =0; i<len; ++i){
	// 	PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
	// 	Py_XDREF(item);
	// }

	free(layer);

	return Py_BuildValue("");
}


PyArrayObject* nlayer_get_a(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->a); 
}
PyArrayObject* nlayer_get_x_hat(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->x_hat); 
}
PyArrayObject* nlayer_get_w(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->w); 
}
PyArrayObject* nlayer_get_c(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->c); 
}
PyArrayObject* nlayer_get_bias(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->bias);
}
PyArrayObject* nlayer_get_psi(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->psi);
}
PyArrayObject* nlayer_get_out(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->out);
}
PyArrayObject* nlayer_get_out_grad(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->out_grad);
}
PyArrayObject* nlayer_get_input(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->input);
}
PyArrayObject* nlayer_get_in_grad(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->in_grad);
}
PyArrayObject* nlayer_get_deda(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->deda);
}
PyArrayObject* nlayer_get_dedw(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dedw);
}
PyArrayObject* nlayer_get_prev_dw(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->prev_dw);
}
PyArrayObject* nlayer_get_dedc(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dedc);
}
PyArrayObject* nlayer_get_prev_dc(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->prev_dc);
}
PyArrayObject* nlayer_get_dedpsi(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dedpsi);
}
PyArrayObject* nlayer_get_dbias(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dbias);
}
PyArrayObject* nlayer_get_prev_dbias(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->prev_dbias);
}
PyArrayObject* nlayer_get_dedinput(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dedinput);
}
PyArrayObject* nlayer_get_dedgradin(PyObject* self, PyObject* args){ 
	PyObject* layer_cap;
	if(!PyArg_ParseTuple(args, "O", &layer_cap)){
		return NULL;
	}
	NLayer* layer = (NLayer*) PyCapsule_GetPointer(layer_cap, "NLayer");
	return matrix_toarray(layer->dedgradin);
}


static npy_double log_eval(npy_double x){
	return 1.0/(1.0 + exp(-x));
}

static npy_double log_deval(npy_double x){
	npy_double s = log_eval(x);
	return s * (1 - s);
}

static npy_double log_ddeval(npy_double x){
	npy_double ex = exp(x);
	npy_double exp1 = (ex+1);
	npy_double ex3 = exp1*exp1*exp1;
	return -ex*(ex-1)/ex3;
}

static npy_double lin_eval(npy_double x){
	return x;
}

static npy_double lin_deval(npy_double x){
	return 1.0;
}

static npy_double lin_ddeval(npy_double x){
	return 0.0;
}

static npy_double rect_eval(npy_double x){
	return (x>0)? x : 0.0;
}

static npy_double rect_deval(npy_double x){
	return (x>0)? 1.0 : 0.0;
}

static npy_double rect_ddeval(npy_double x){
	return 0.0;
}

static npy_double rbf_eval(npy_double x){
	return exp(-x);
}

static npy_double rbf_deval(npy_double x){
	return -exp(-x);
}

static npy_double rbf_ddeval(npy_double x){
	return exp(-x);
}




static PyObject* get_logistic_sig(PyObject* self, PyObject* args){
	uint i;
	if(!PyArg_ParseTuple(args, "I", &i)){
		return NULL;
	}
	assert(i<3 && i>=0);
	sig_fun p = NULL;
	switch(i){
		case 0:
			p = &log_eval;
			break;
		case 1:
			p = &log_deval;
			break;
		case 2:
			p = &log_ddeval;
			break;
	}
	return PyCapsule_New(p, "function_pointer", NULL);
} 

static PyObject* get_linear_sig(PyObject* self, PyObject* args){
	uint i;
	if(!PyArg_ParseTuple(args, "I", &i)){
		return NULL;
	}
	assert(i<3 && i>=0);
	sig_fun p = NULL;
	switch(i){
		case 0:
			p = &lin_eval;
			break;
		case 1:
			p = &lin_deval;
			break;
		case 2:
			p = &lin_ddeval;
			break;
	}
	return PyCapsule_New(p, "function_pointer", NULL);
} 

static PyObject* get_rect_sig(PyObject* self, PyObject* args){
	uint i;
	if(!PyArg_ParseTuple(args, "I", &i)){
		return NULL;
	}

	assert(i<3 && i>=0);
	sig_fun p = NULL;
	switch(i){
		case 0:
			p = rect_eval;
			break;
		case 1:
			p = rect_deval;
			break;
		case 2:
			p = rect_ddeval;
			break;
	}
	
	return PyCapsule_New(p, "function_pointer", NULL);
}

static PyObject* get_rbf_sig(PyObject* self, PyObject* args){
	uint i;
	if(!PyArg_ParseTuple(args, "I", &i)){
		return NULL;
	}

	assert(i<3 && i>=0);
	sig_fun p = NULL;
	switch(i){
		case 0:
			p = rbf_eval;
			break;
		case 1:
			p = rbf_deval;
			break;
		case 2:
			p = rbf_ddeval;
			break;
	}
	
	return PyCapsule_New(p, "function_pointer", NULL);
}

static char generic_doc[] = "returns an array view of the field";
static char generic_nothing[] = "doc missing...";


static PyMethodDef ext_neuro_methods[] = {
	{"create_layer", create_layer, METH_VARARGS, generic_nothing},
	{"compute_gradient_from_np", compute_gradient_from_np, METH_VARARGS, generic_nothing},
	{"get_logistic_sig", get_logistic_sig, METH_VARARGS, generic_nothing},
	{"get_linear_sig", get_linear_sig, METH_VARARGS, generic_nothing},
	{"get_rect_sig", get_rect_sig, METH_VARARGS, generic_nothing},
	{"get_rbf_sig", get_rect_sig, METH_VARARGS, generic_nothing},
	{"get_a", nlayer_get_a, METH_VARARGS, generic_doc},
	{"get_w", nlayer_get_w, METH_VARARGS, generic_doc},
	{"get_x_hat", nlayer_get_x_hat, METH_VARARGS, generic_doc},
	{"get_c", nlayer_get_c, METH_VARARGS, generic_doc},
	{"get_bias", nlayer_get_bias, METH_VARARGS, generic_doc},
	{"get_psi", nlayer_get_psi, METH_VARARGS, generic_doc},
	{"get_out", nlayer_get_out, METH_VARARGS, generic_doc},
	{"get_out_grad", nlayer_get_out_grad, METH_VARARGS, generic_doc},
	{"get_input", nlayer_get_input, METH_VARARGS, generic_doc},
	{"get_in_grad", nlayer_get_in_grad, METH_VARARGS, generic_doc},
	{"get_deda", nlayer_get_deda, METH_VARARGS, generic_doc},
	{"get_dedw", nlayer_get_dedw, METH_VARARGS, generic_doc},
	{"get_prev_dw", nlayer_get_prev_dw, METH_VARARGS, generic_doc},
	{"get_dedc", nlayer_get_dedc, METH_VARARGS, generic_doc},
	{"get_prev_dc", nlayer_get_prev_dc, METH_VARARGS, generic_doc},
	{"get_dedpsi", nlayer_get_dedpsi, METH_VARARGS, generic_doc},
	{"get_dbias", nlayer_get_dbias, METH_VARARGS, generic_doc},
	{"get_prev_dbias", nlayer_get_prev_dbias, METH_VARARGS, generic_doc},
	{"get_dedinput", nlayer_get_dedinput, METH_VARARGS, generic_doc},
	{"get_dedgradin", nlayer_get_dedgradin, METH_VARARGS, generic_doc},
	{"evaluate_layer_from_np", evaluate_layer_from_np, METH_VARARGS, generic_nothing},
	{"update_weights_from_py", update_weights_from_py, METH_VARARGS, generic_nothing},
	{"destroy_layer", destroy_layer, METH_VARARGS, generic_nothing},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initext_neuro()
{
	Py_InitModule("ext_neuro", ext_neuro_methods);
	import_array(); /* required NumPy initialization */
}