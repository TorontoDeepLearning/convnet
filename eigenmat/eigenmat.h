#ifndef EIGENMAT_H
#define EIGENMAT_H

#include "common.h"

#include <random>

#define ERROR_INCOMPATIBLE_DIMENSIONS -1
#define CUBLAS_ERROR -2
#define CUDA_ERROR -3
#define VIEW_ERROR -4
#define ERROR_TRANSPOSED -5
#define ERROR_GENERIC -6
#define ERROR_TRANSPOSEDNESS -7
#define ERROR_NOT_ON_DEVICE -8
#define ERROR_UNSUPPORTED -9

struct eigenmat
{
  float* data;
  int size[2];
  int is_trans; // 0 or 1
  int owns_data;
};

struct rnd_struct_e
{
    rnd_struct_e() : pGenerator(NULL) {}
    ~rnd_struct_e() { if (pGenerator) { delete pGenerator; } }

    std::default_random_engine *pGenerator;
};

int get_leading_dimension(eigenmat* mat);
int get_nonleading_dimension(eigenmat* mat);
void set_transpose(eigenmat* mat, int is_trans);
inline char get_transpose_char(eigenmat* mat);
int allocate_memory(eigenmat* mat);
int cumsum_by_axis(eigenmat* mat, eigenmat* target, int axis);
int selectCols(eigenmat* source, eigenmat* target, eigenmat* indices);
int swapRows(eigenmat* source, eigenmat* target, eigenmat* indices1, eigenmat* indices2);
int setSelectedCols(eigenmat* source, eigenmat* target, eigenmat* indices);
void sgemm(bool rowMajor, bool TransA, bool TransB, int M, int N, int K, float alpha, float* A, int lda, float* B, int ldb, float beta, float* C, int ldc);

int init_random(rnd_struct_e* rnd_state, int seed);
float uniform(rnd_struct_e* rnd_state);
float normal(rnd_struct_e* rnd_state);
int fill_with_rand(rnd_struct_e* rnd_state, eigenmat* mat);
int fill_with_randn(rnd_struct_e* rnd_state, eigenmat* mat);
int sample_bernoulli(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target);
int sample_bernoulli_tanh(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target);
int sample_gaussian(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target, float mult);
int perturb_energy(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target);
int perturb_prob(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target);
int dropout(rnd_struct_e* rnd_state, eigenmat* mat, float dropprob, float val, float scale);

int write_at(eigenmat* mat, int row, int col, float val);
float read_from(eigenmat* mat, int row, int col, int* err_code);
int copy_on_device(eigenmat* mat1, eigenmat* mat2);
int get_row_slice(eigenmat* source, eigenmat* target, unsigned int start, unsigned int end);
int set_row_slice(eigenmat* source, eigenmat* target, unsigned int start, unsigned int end);
int copy_transpose(eigenmat* source, eigenmat* target);
int set_shape(eigenmat* mat, unsigned int m, unsigned int n);
int reshape(eigenmat* mat, int m, int n);
int get_slice(eigenmat* source, eigenmat* target, unsigned int first_col, unsigned int last_col);
int get_vector_slice(eigenmat* source, eigenmat* target, unsigned int first_ind, unsigned int last_ind);
void init_from_array(eigenmat* mat, float* data, int m, int n);
int init_empty(eigenmat* mat, int m, int n);

int add_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int add_col_mult(eigenmat* mat, eigenmat* vec, eigenmat* target, float mult);
int add_to_each_pixel(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float mult);
int mult_diagonal_scalar(eigenmat* mat, float val, eigenmat* target);
int add_diagonal_scalar(eigenmat* mat, float val, eigenmat* target);
int mult_diagonal(eigenmat* mat, eigenmat* vec, eigenmat* target);
int add_diagonal(eigenmat* mat, eigenmat* vec, eigenmat* target);
int add_row_mult(eigenmat* mat, eigenmat* vec, eigenmat* target, float mult);
int add_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int mult_by_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int mult_by_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int div_by_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int div_by_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target);
int less_than(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int less_than_scalar(eigenmat* mat, float val, eigenmat* target);
int greater_than(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int greater_than_scalar(eigenmat* mat, float val, eigenmat* target);
int upper_bound(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int lower_bound(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int upper_bound_mod_scalar(eigenmat* mat, float val, eigenmat* target);
int upper_bound_scalar(eigenmat* mat, float val, eigenmat* target);
int lower_bound_scalar(eigenmat* mat, float val, eigenmat* target);
int max_by_axis(eigenmat* mat, eigenmat* target, int axis);
int choose_max_and_accumulate(eigenmat* mat, eigenmat* acc);
int choose_max_by_axis(eigenmat* mat, eigenmat* target, int axis);
int argmax_by_axis(eigenmat* mat, eigenmat* target, int axis);
int sqsum_by_axis(eigenmat* mat, eigenmat* target, int axis, float mult, float p);
int sum_by_axis(eigenmat* mat, eigenmat* target, int axis, float mult, float p);
float sum_all(eigenmat* mat);
int normlimit_by_axis(eigenmat* mat, eigenmat* target, int axis, float norm, int constraint);
int normalize_by_axis(eigenmat* mat, eigenmat* target, int axis);
int sign(eigenmat* mat, eigenmat* target);
int apply_cos(eigenmat* mat, eigenmat* target);
int apply_sin(eigenmat* mat, eigenmat* target);
int apply_sigmoid(eigenmat* mat, eigenmat* target);
int apply_tanh(eigenmat* mat, eigenmat* target);
int apply_abs(eigenmat* mat, eigenmat* target);
int apply_log_1_plus_exp(eigenmat* mat, eigenmat* target);
int apply_relu_squash(eigenmat* mat, eigenmat* target, float lambda);
int apply_log(eigenmat* mat, eigenmat* target, float tiny);
int apply_exp(eigenmat* mat, eigenmat* target);
int apply_ceil(eigenmat* mat, eigenmat* target);
int apply_floor(eigenmat* mat, eigenmat* target);
int apply_sqrt(eigenmat* mat, eigenmat* target);
int apply_pow(eigenmat* mat, float exponent, eigenmat* target);
int apply_pow_matrix(eigenmat* mat, eigenmat* exponent, eigenmat* target);
int compute_cross_entropy(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float tiny);
int compute_cross_entropy_bernoulli(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float tiny);
int correct_preds(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float cutoff);
int reciprocal(eigenmat* mat, eigenmat* target);
int dot(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float beta, float alpha);
float vdot(eigenmat* mat1, eigenmat* mat2, int* err_code);
int add_mult(eigenmat* mat1, eigenmat* mat2, float alpha);
int add_mult_sign(eigenmat* mat, eigenmat* mat2, float mult);
int add_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int subtract_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int divide_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int mult_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_sin_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_cos_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_logistic_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_tanh_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_rectified_linear_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int apply_rectified_linear_smooth_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target);
int assign_scalar(eigenmat* mat, float alpha);
int mult_by_scalar(eigenmat* mat, float alpha, eigenmat* target);
int divide_by_scalar(eigenmat* mat, float alpha, eigenmat* target);
int add_scalar(eigenmat* mat, float alpha, eigenmat* target);
float euclid_norm(eigenmat* mat);
int selectRows(eigenmat* source, eigenmat* target, eigenmat* indices);
int swapColumns(eigenmat* source, eigenmat* target, eigenmat* indices1, eigenmat* indices2);
int shuffleColumns(eigenmat* source, eigenmat* rand_perm_indices);
int setSelectedRows(eigenmat* source, eigenmat* target, eigenmat* indices);
int blockify(eigenmat* source, eigenmat* target, int blocksize);
int softmax(eigenmat* mat, eigenmat* target);
int softmax_row_major(eigenmat* mat);
int softmax_row_major_multi(eigenmat* mat, int numslices);
int apply_logistic_grad(eigenmat* mat1, eigenmat* mat2, eigenmat* out_grad);
int apply_softmax_grad(eigenmat* mat, eigenmat* labels, eigenmat* target);
int apply_softmax_grad_row_major(eigenmat* mat, eigenmat* labels, eigenmat* target);
int get_softmax_correct(eigenmat* mat, eigenmat* labels, eigenmat* target);
int get_softmax_correct_row_major(eigenmat* mat, eigenmat* labels, eigenmat* target);
int hinge_loss_row_major(eigenmat* mat, eigenmat* labels, eigenmat* target, int quadratic, float margin);
int get_softmax_cross_entropy(eigenmat* mat, eigenmat* labels, eigenmat* target, float tiny);
int get_softmax_cross_entropy_row_major(eigenmat* mat, eigenmat* labels, eigenmat* target, float tiny);
int extract_patches(eigenmat* images, eigenmat* patches, eigenmat* width_offset,
                    eigenmat* height_offset, eigenmat* flip, int img_width,
                    int img_height, int patch_width, int patch_height);
int rectify_bounding_boxes(eigenmat* boxes, eigenmat* width_offset,
                           eigenmat* height_offset, eigenmat* flip,
                           int patch_width, int patch_height);
int adagrad(eigenmat* history, eigenmat* grad, float delta);
int rms_prop(eigenmat* history, eigenmat* grad, float factor);
int get_logistic_correct_normalized(eigenmat* mat1, eigenmat* mat2, eigenmat* out);

// LSTMs
int lstm_fprop(eigenmat* s_in, eigenmat* s_out, eigenmat* w_dense, eigenmat* w_diag, eigenmat* b, bool init, bool use_relu);
int lstm_bprop(eigenmat* s_in, eigenmat* s_out, eigenmat* d_in, eigenmat* d_out, eigenmat* w_dense, eigenmat* w_diag, bool init, bool use_relu);
int lstm_outp(eigenmat* s_in, eigenmat* s_out, eigenmat* d_out, eigenmat* dw_dense, eigenmat* dw_diag, eigenmat* db, bool init);

int bn_bprop(eigenmat* deriv, eigenmat* input, eigenmat* gamma, eigenmat* mu,
             eigenmat* sigma, eigenmat* target, float scale_targets);
int bn_grad(eigenmat* deriv, eigenmat* input, eigenmat* mu, eigenmat* sigma,
            eigenmat* dgamma, eigenmat* dbeta);
#endif

