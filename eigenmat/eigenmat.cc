#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>

#include "eigenmat.h"

using namespace std;

/* ------------------------------ Utility routines ------------------------------ */

int get_leading_dimension(eigenmat* mat) {
  return mat->is_trans ? mat->size[1] : mat->size[0];
}

int get_nonleading_dimension(eigenmat* mat) {
  return mat->is_trans ? mat->size[0] : mat->size[1];
}

void set_transpose(eigenmat* mat, int is_trans) {
  mat->is_trans = is_trans;
}

inline char get_transpose_char(eigenmat* mat) {
  return mat->is_trans ? 't' : 'n';
}

/* ------------------------------ Allocating/moving data ------------------------------ */

int allocate_memory(eigenmat* mat) {
  const int len = mat->size[0] * mat->size[1];
  mat->data = (float*)malloc(len * sizeof(float));
  return 0;
}

int copy_on_device(eigenmat* mat1, eigenmat* mat2) {
  const int len = mat1->size[0]*mat1->size[1];

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::VectorXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::VectorXf> eig_mat2(mat2->data, len);
  eig_mat2 = eig_mat1;

  return 0;
}

int get_row_slice(eigenmat* source, eigenmat* target, unsigned int start, unsigned int end) {
  int height = source->size[0];
  int width = source->size[1];

  if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  return 0;
}

int set_row_slice(eigenmat* source, eigenmat* target, unsigned int start, unsigned int end) {
  int height = target->size[0];
  int width = target->size[1];

  if ((end - start) != source->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  return 0;
}

int copy_transpose(eigenmat* source, eigenmat* target) {
  unsigned int height = source->size[0];
  unsigned int width = source->size[1];

  if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::MatrixXf> eig_source(source->data, source->size[0], source->size[1]);
  Eigen::Map<Eigen::MatrixXf> eig_target(target->data, target->size[0], target->size[1]);
  eig_target = eig_source.transpose();
  return 0;
}

int set_shape(eigenmat* mat, unsigned int m, unsigned int n) {
  mat->size[0] = m;
  mat->size[1] = n;
  return 0;
}

int reshape(eigenmat* mat, int m, int n) {
    if (m < 0 && n < 0)
        return ERROR_GENERIC;

    if (m < 0)
        m = (mat->size[0] * mat->size[1]) / n;
    if (n < 0)
        n = (mat->size[0] * mat->size[1]) / m;

    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

int get_slice(eigenmat* source, eigenmat* target, unsigned int first_col, unsigned int last_col) {
  if (source->is_trans)
    return ERROR_TRANSPOSED;

  if (last_col > source->size[1] || (first_col >= last_col))
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  int num_rows = source->size[0];

  target->data = source->data + first_col * num_rows;
  target->size[0] = source->size[0];
  target->size[1] = last_col - first_col;
  target->is_trans = 0;
  target->owns_data = 0;

  return 0;
}

int get_vector_slice(eigenmat* source, eigenmat* target, unsigned int first_ind, unsigned int last_ind) {
  // source must be a vector
  if (source->size[0] > 1 && source->size[1] > 1)
    return ERROR_GENERIC;

  if (source->is_trans)
    return ERROR_TRANSPOSED;

  if (first_ind >= last_ind)
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  int num_rows = source->size[0];

  target->data = source->data + first_ind * num_rows;
  target->is_trans = 0;
  target->owns_data = 0;

  if (source->size[0] > 1) {
    if (last_ind > source->size[0])
      return ERROR_INCOMPATIBLE_DIMENSIONS;

    target->size[0] = last_ind - first_ind;
    target->size[1] = 1;
  } else {
    if (last_ind > source->size[1])
      return ERROR_INCOMPATIBLE_DIMENSIONS;

    target->size[0] = 1;
    target->size[1] = last_ind - first_ind;
  }

  return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

void init_from_array(eigenmat* mat, float* data, int m, int n) {
  mat->data = data;
  mat->size[0] = m;
  mat->size[1] = n;
  mat->is_trans = 0;
  mat->owns_data = 1;
}

int init_empty(eigenmat* mat, int m, int n) {
  mat->size[0] = m;
  mat->size[1] = n;
  mat->is_trans = 0;
  mat->owns_data = 1;

  return allocate_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */

int init_random(rnd_struct_e* rnd_state, int seed)
{
    if (rnd_state->pGenerator)
    {
        delete rnd_state->pGenerator;
    }
    rnd_state->pGenerator = new default_random_engine(seed);
    return 0;
}

float uniform(rnd_struct_e* rnd_state)
{
    uniform_real_distribution<float> distribution(0, 1);
    return distribution(*rnd_state->pGenerator);
}

float normal(rnd_struct_e* rnd_state)
{
    normal_distribution<float> distribution(0, 1);
    return distribution(*rnd_state->pGenerator);
}

int fill_with_rand(rnd_struct_e* rnd_state, eigenmat* mat) {
  const int len = mat->size[0] * mat->size[1];
  for (int i = 0; i < len; i++)
    mat->data[i] = uniform(rnd_state); 
  return 0;
}

int fill_with_randn(rnd_struct_e* rnd_state, eigenmat* mat) {
  const int len = mat->size[0] * mat->size[1];
  for (int i = 0; i < len; i++)
    mat->data[i] = normal(rnd_state);
  return 0;
}

int sample_bernoulli(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target) {
  int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] > uniform(rnd_state) ? 1 : 0; 

  return 0;
}

int sample_bernoulli_tanh(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target) {
  int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] > uniform(rnd_state) ? 1 : -1; 

  return 0;
}

int sample_gaussian(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target, float mult) {
  int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] + mult * normal(rnd_state);

  return 0;
}

int perturb_energy(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target) {
  const int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] - log(-log(uniform(rnd_state)));

  return 0;
}

int perturb_prob(rnd_struct_e* rnd_state, eigenmat* mat, eigenmat* target) {
  const int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = -mat->data[i] / log(uniform(rnd_state));

  return 0;
}

int dropout(rnd_struct_e* rnd_state, eigenmat* mat, float dropprob, float val, float scale)
{
    const int len = mat->size[0] * mat->size[1];
    for (int i=0; i<len; i++)
    {
        if (dropprob > uniform(rnd_state))
        {
            mat->data[i] = val;
        } else
        {
            mat->data[i] *= scale;
        }
    }

    return 0;
}

/* ------------------------------ Algebraic operations ------------------------------ */

int add_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target)
{
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, h);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.colwise() + eig_vec;

  return 0;
}

int add_mult_sign(eigenmat* mat, eigenmat* mat2, float mult) {
  for (int i = 0; i < mat->size[0] * mat->size[1]; i++) {
    mat->data[i] += (mat2->data[i] == 0) ? 0 : ((mat2->data[i] > 0) ? mult:-mult);
  }
  return 0;
}

int add_col_mult(eigenmat* mat, eigenmat* vec, eigenmat* target, float mult)
{
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, h);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.colwise() + eig_vec * mult;

  return 0;
}

int mult_diagonal_scalar(eigenmat* mat, float val, eigenmat* target) {
  unsigned int w = mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < w; i++) target->data[i*w+i] = val * mat->data[i*w+i];

  return 0;
}

int add_diagonal_scalar(eigenmat* mat, float val, eigenmat* target) {
  unsigned int w = mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < w; i++) target->data[i*w+i] = val * mat->data[i*w+i];

  return 0;
}

int mult_diagonal(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int w = mat->size[1];

  if (mat->size[0] != vec->size[1] * vec->size[0] ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < w; i++) target->data[i*w+i] = vec->data[i] * mat->data[i*w+i];

  return 0;
}

int add_diagonal(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int w = mat->size[1];

  if (mat->size[0] != vec->size[1] * vec->size[0] ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < w; i++) target->data[i*w+i] = vec->data[i] + mat->data[i*w+i];
  return 0;
}

int add_row_mult(eigenmat* mat, eigenmat* vec, eigenmat* target, float mult) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, w);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.rowwise() + eig_vec.transpose() * mult;
  return 0;
}

int add_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, w);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.rowwise() + eig_vec.transpose();
  return 0;
}

int mult_by_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, h);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.colwise() * eig_vec;

  return 0;
}

int mult_by_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, w);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.rowwise() * eig_vec.transpose();
  return 0;
}

int div_by_col_vec(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, h);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.colwise() / eig_vec;
  return 0;
}

int div_by_row_vec(eigenmat* mat, eigenmat* vec, eigenmat* target) {
  unsigned int h = mat->size[0];
  unsigned int w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
    mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, h, w);
  Eigen::Map<Eigen::ArrayXf> eig_vec(vec->data, w);
  Eigen::Map<Eigen::ArrayXXf> eig_target(target->data, h, w);

  eig_target = eig_mat.rowwise() / eig_vec.transpose();

  return 0;
}

int less_than(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  const int len = mat1->size[0] * mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat1->data[i] < mat2->data[i] ? 1 : 0;
  return 0;
}

int less_than_scalar(eigenmat* mat, float val, eigenmat* target) {
  const int len = mat->size[0]*mat->size[1];

  if (mat->is_trans != target->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] < val ? 1 : 0;
  
  return 0;
}

int greater_than(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat1->data[i] > mat2->data[i] ? 1 : 0;

  return 0;
}

int upper_bound(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat1->data[i] < mat2->data[i] ? mat1->data[i] : mat2->data[i];

  return 0;
}

int lower_bound(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat1->data[i] > mat2->data[i] ? mat1->data[i] : mat2->data[i];
}

int greater_than_scalar(eigenmat* mat, float val, eigenmat* target) {
  int len = mat->size[0]*mat->size[1];

  if (mat->is_trans != target->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] > val ? 1:0;
  return 0;
}

int upper_bound_mod_scalar(eigenmat* mat, float val, eigenmat* target)
{
    int len = mat->size[0]*mat->size[1];

    if (mat->is_trans != target->is_trans)
      return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
      return ERROR_INCOMPATIBLE_DIMENSIONS;

    for (int i=0; i<len; i++)
    {
        float curr = mat->data[i];
        if (curr > val)
        {
            target->data[i] = val;
        } else
        if (curr < -val)
        {
            target->data[i] = -val;
        } else
        {
            target->data[i] = curr;
        }
    }

    return 0;
}

int upper_bound_scalar(eigenmat* mat, float val, eigenmat* target) {
  int len = mat->size[0]*mat->size[1];

  if (mat->is_trans != target->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] < val ? mat->data[i] : val;

  return 0;
}

int lower_bound_scalar(eigenmat* mat, float val, eigenmat* target) {
  const int len = mat->size[0]*mat->size[1];

  if (mat->is_trans != target->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < len; i++)
    target->data[i] = mat->data[i] > val ? mat->data[i] : val;

  return 0;
}

int cumsum_by_axis(eigenmat* mat, eigenmat* target, int axis) {
  const unsigned int h = mat->size[0], w = mat->size[1];
  if (axis == 0) {
    for (int i = 0; i < w; i++) {
      const float *mat_data = &mat->data[i * h];
      float *target_data = &target->data[i * h];
      target_data[0] = mat_data[0];
      for (int j = 1; j < h; j++) 
        target_data[j] =  target_data[j-1] + mat_data[j];
    }
  } else if (axis == 1) {
    for (int i = 0; i < h; i++) {
      const float *mat_data = &mat->data[i];
      float *target_data = &target->data[i];
      target_data[0] = mat_data[0];
      for (int j = 1; j < w; j++) 
        target_data[j*h] =  target_data[(j-1)*h] + mat_data[j*h];
    }
  } else {
    return ERROR_UNSUPPORTED;
  }
  return 0;
}

int max_by_axis(eigenmat* mat, eigenmat* target, int axis) {
  const unsigned int h = mat->size[0], w = mat->size[1];
  if (axis == 0) {
    for (int i = 0; i < w; i++) {
      const float *mat_data = &mat->data[i * h];
      float max = mat_data[0];
      for (int j = 1; j < h; j++) 
        if (max < mat_data[j]) max = mat_data[j];
      target->data[i] = max;
    }
  } else if (axis == 1) {
    for (int i = 0; i < h; i++) {
      const float *mat_data = &mat->data[i];
      float max = mat_data[0];
      for (int j = 1; j < w; j++) 
        if (max < mat_data[j*h]) max = mat_data[j*h];
      target->data[i] = max;
    }
  } else {
    return ERROR_UNSUPPORTED;
  }
  return 0;
}

int choose_max_and_accumulate(eigenmat* mat, eigenmat* acc) {
  const unsigned int h = mat->size[0], w = mat->size[1];

  if (mat->is_trans)
    return ERROR_TRANSPOSED;

  if (acc->size[0] != mat->size[0] || acc->size[1] != mat->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for (int i = 0; i < w; i++) {
    int argmax = 0;
    const float *mat_data = &mat->data[i * h];
    for (int j = 1; j < h; j++) 
      if (mat_data[argmax] < mat_data[j]) argmax = j;
    acc->data[i * h + argmax] += 1;
  }

  return 0;
}

int choose_max_by_axis(eigenmat* mat, eigenmat* target, int axis) {
  const unsigned int h = mat->size[0], w = mat->size[1];
  if (axis == 0) {
    for (int i = 0; i < w; i++) {
      int argmax = 0;
      const float *mat_data = &mat->data[i * h];
      float *target_data = &target->data[i * h];
      for (int j = 1; j < h; j++) 
        if (mat_data[argmax] < mat_data[j]) argmax = j;
      for (int j = 0; j < h; j++) target_data[j] = (j == argmax) ? 1 : 0;
    }
  } else if (axis == 1) {
    for (int i = 0; i < h; i++) {
      int argmax = 0;
      const float *mat_data = &mat->data[i];
      float *target_data = &target->data[i];
      for (int j = 1; j < w; j++) 
        if (mat_data[argmax*h] < mat_data[j*h]) argmax = j;
      for (int j = 0; j < w; j++) target_data[j*h] = (j == argmax) ? 1 : 0;
    }
  } else {
    return ERROR_UNSUPPORTED;
  }
  return 0;
}

int argmax_by_axis(eigenmat* mat, eigenmat* target, int axis) {
  const unsigned int h = mat->size[0], w = mat->size[1];
  if (axis == 0) {
    for (int i = 0; i < w; i++) {
      int argmax = 0;
      const float *mat_data = &mat->data[i * h];
      for (int j = 1; j < h; j++) 
        if (mat_data[argmax] < mat_data[j]) argmax = j;
      target->data[i] = argmax;
    }
  } else if (axis == 1) {
    for (int i = 0; i < h; i++) {
      int argmax = 0;
      const float *mat_data = &mat->data[i];
      for (int j = 1; j < w; j++) 
        if (mat_data[argmax*h] < mat_data[j*h]) argmax = j;
      target->data[i] = argmax;
    }
  } else {
    return ERROR_UNSUPPORTED;
  }
  return 0;
}

int sqsum_by_axis(eigenmat* mat, eigenmat* target, int axis, float mult, float p) {
  const unsigned int h = mat->size[0];
  const unsigned int w = mat->size[1];

  if (axis == 0) {
    for (int i = 0; i < w; i++) {
      float sum = 0;
      const float *mat_data = &mat->data[i * h];
      for (int j = 0; j < h; j++)
        sum += mat_data[j] * mat_data[j];
      target->data[i] = p * target->data[i] + mult * sum;
    }
  } else
  if (axis == 1) {
    for (int i = 0; i < h; i++) {
      float sum = 0;
      const float *mat_data = &mat->data[i];
      for (int j = 0; j < w; j++)
        sum += mat_data[j*h] * mat_data[j*h];
      target->data[i] = p * target->data[i] + mult * sum;
    }
  } else {
    return ERROR_UNSUPPORTED;
  }
  return 0;
}

int sum_by_axis(eigenmat* mat, eigenmat* target, int axis, float mult, float p) {
  int len_mat = mat->size[0] * mat->size[1];
  int len_target = target->size[0] * target->size[1];
  Eigen::Map<Eigen::ArrayXXf> eig_mat(mat->data, mat->size[0], mat->size[1]);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len_target);
  if (mat->is_trans)
    axis = 1 - axis;
  if (axis == 0) {
    eig_target = p * eig_target + mult * eig_mat.colwise().sum();
  } else {
    eig_target = p * eig_target + mult * eig_mat.rowwise().sum();
  }
  return 0;
}

int sign(eigenmat* mat, eigenmat* target) {
  int len = mat->size[0]*mat->size[1];

  if (mat->is_trans != target->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < len; i++)
    target->data[i] = (mat->data[i] < 0) ? -1 : (mat->data[i] > 0 ? 1 : 0);

  return 0;
}

int apply_cos(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.cos();
  return 0;
}

int apply_sin(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.sin();

  return 0;
}

int apply_softmax(eigenmat* mat, eigenmat* target)
{
    int width = mat->size[1];
    int height = mat->size[0];

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    #pragma omp parallel for
    for (int i=0; i<width; i++)
    {
        float *mat_data = &mat->data[i * height];
        float *target_data = &target->data[i * height];
        float max = mat_data[0], sum = 0;
        for (int j=1; j<height; j++)
        {
            if (max < mat_data[j])
            {
                max = mat_data[j];
            }
        }

        for (int j=0; j<height; j++)
        {
            target_data[j] = exp(mat_data[j] - max);
            sum += target_data[j];
        }

        for (int j=0; j<height; j++)
        {
            target_data[j] /= sum;
        }
    }

    return 0;
}

int apply_softmax_row_major(eigenmat* mat, eigenmat* target)
{
    unsigned int width = mat->size[1];
    unsigned int height = mat->size[0];

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    #pragma omp parallel for
    for (int i=0; i<height; i++)
    {
        float max = mat->data[i], sum = 0;
        for (int j=1; j<width; j++)
        {
            float curr = mat->data[j * height + i];
            if (max < curr)
            {
                max = curr;
            }
        }

        for (int j=0; j<width; j++)
        {
            int idx = j * height + i;
            target->data[idx] = exp(mat->data[idx] - max);
            sum += target->data[idx];
        }
        for (int j=0; j<width; j++)
        {
            target->data[j * height + i] /= sum;
        }
    }

    return 0;
}

int apply_softmax_grad(eigenmat* mat, eigenmat* labels, eigenmat* target) {
  const int width = mat->size[1], height = mat->size[0];

  if (target != mat) {
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) target->data[i] = mat->data[i];
  }
  for (int i = 0; i < width; i++) {
    target->data[i * height + (int)labels->data[i]] -= 1.0;
  }
  return 0;
}

int get_softmax_cross_entropy(eigenmat* mat, eigenmat* labels, eigenmat* target, const float tiny) {
  const int width = mat->size[1], height = mat->size[0];

  for (int i = 0; i < width; i++) {
    target->data[i] = -log(mat->data[i * height + (int)labels->data[i]]);
  }
  return 0;
}

int get_softmax_correct(eigenmat* mat, eigenmat* labels, eigenmat* target) {
  int width = mat->size[1], height = mat->size[0];

  #pragma omp parallel for
  for (int i = 0; i < width; i++) {
    int argmax = 0;
    const int correct_label = (int)labels->data[i];
    float *mat_data = &mat->data[i * height];
    for (int j = 1; j < height; j++) if (mat_data[argmax] < mat_data[j]) argmax = j;
    target->data[i] = (correct_label == argmax) ? 1: 0;
  }
  return 0;
}

float sum_all(eigenmat* mat) {
  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, mat->size[0] * mat->size[1]);
  return eig_mat.sum();
}

int apply_sigmoid(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for(int i = 0; i < len; i++) target->data[i] = 1 / (1 + exp(-mat->data[i]));
  return 0;
}

int apply_tanh(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  #pragma omp parallel for
  for(int i = 0; i < len; i++) target->data[i] = 2 / (1 + exp(-mat->data[i])) - 1;
  return 0;
}

int apply_abs(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.abs();
  return 0;
}

int apply_log_1_plus_exp(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  for(int i = 0; i < len; i++) target->data[i] = log(1+exp(mat->data[i]));

  return 0;
}

int apply_log(eigenmat* mat, eigenmat* target, float tiny) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.log();

  return 0;
}

int apply_exp(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.exp();

  return 0;
}

int apply_ceil(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  
  #pragma omp parallel for
  for(int i = 0; i < len; i++) target->data[i] = ceil(mat->data[i]);

  return 0;
}

int apply_floor(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for(int i = 0; i < len; i++) target->data[i] = floor(mat->data[i]);

  return 0;
}

int apply_sqrt(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.sqrt();

  return 0;
}

int apply_pow(eigenmat* mat, float exponent, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat.pow(exponent);

  return 0;
}

int apply_pow_matrix(eigenmat* mat, eigenmat* exponent, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (mat->size[0] != exponent->size[0] || mat->size[1] != exponent->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for(int i = 0; i < len; i++) target->data[i] = pow(mat->data[i], exponent->data[i]);
  return 0;
}

int compute_cross_entropy(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float tiny) {
  unsigned int len = mat1->size[0] * mat1->size[1];

  if (mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (mat2->size[0] != mat2->size[0] || mat2->size[1] != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++) target->data[i] = - mat1->data[i] * log(mat2->data[i] + tiny);

  return 0;
}

int compute_cross_entropy_bernoulli(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float tiny) {
  unsigned int len = mat1->size[0] * mat1->size[1];

  if (mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = - mat1->data[i] * log(mat2->data[i] + tiny) - (1-mat1->data[i]) * log(1 - mat2->data[i] + tiny);
  return 0;
}

int correct_preds(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float cutoff) {
  unsigned int len = mat1->size[0] * mat1->size[1];

  if (mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++)
    target->data[i] = mat1->data[i] * mat2->data[i] >= cutoff + (1 - mat1->data[i]) * (mat2->data[i] < cutoff);
  return 0;
}

int reciprocal(eigenmat* mat, eigenmat* target) {
  unsigned int len = mat->size[0] * mat->size[1];

  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  
  eig_target = eig_mat.inverse();
  
  return 0;
}

int dot(eigenmat* mat1, eigenmat* mat2, eigenmat* target, float beta, float alpha) {

  if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
    get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
    get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  }
  int m = get_leading_dimension(mat1),
    k = get_leading_dimension(mat2),
    n = get_nonleading_dimension(mat2);

  Eigen::Map<Eigen::MatrixXf> eig_mat1(mat1->data, mat1->size[0], mat1->size[1]);
  Eigen::Map<Eigen::MatrixXf> eig_mat2(mat2->data, mat2->size[0], mat2->size[1]);
  Eigen::Map<Eigen::MatrixXf> eig_target(target->data, target->size[0], target->size[1]);

  eig_target = beta * eig_target;
  if (mat1->is_trans && mat2->is_trans) {
    eig_target.noalias() += alpha * (eig_mat1.transpose() * eig_mat2.transpose());
  } else if (mat1->is_trans) {
    eig_target.noalias() += alpha * (eig_mat1.transpose() * eig_mat2);
  } else if (mat2->is_trans) {
    eig_target.noalias() += alpha * (eig_mat1 * eig_mat2.transpose());
  } else {
    eig_target.noalias() += alpha * (eig_mat1 * eig_mat2);
  }
  
  return 0;
}

float vdot(eigenmat* mat1, eigenmat* mat2, int* err_code) {
  int len = mat1->size[0]*mat1->size[1];
  float res;

  if (mat1->is_trans != mat2->is_trans) {
    *err_code = ERROR_TRANSPOSEDNESS;
    return 0;
  }

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
    *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
    return 0;
  }
  Eigen::Map<Eigen::VectorXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::VectorXf> eig_mat2(mat2->data, len);
  return eig_mat1.dot(eig_mat2);
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
  have the same transposedness. */
int add_mult(eigenmat* mat1, eigenmat* mat2, float alpha) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_mat2(mat2->data, len);
  eig_mat1 += eig_mat2 * alpha;
  return 0;
}

int add_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_mat2(mat2->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat1 + eig_mat2 ;
  return 0;
}

int subtract_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_mat2(mat2->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat1 - eig_mat2 ;
  return 0;
}

int divide_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::ArrayXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_mat2(mat2->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat1 / eig_mat2 ;

  return 0;
}

/* Elementwise multiplication of 2 matrices */
int mult_elementwise(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  Eigen::Map<Eigen::ArrayXf> eig_mat1(mat1->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_mat2(mat2->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat1 * eig_mat2 ;
  return 0;
}

int apply_sin_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  return 0;
}

int apply_cos_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  return 0;
}

int apply_logistic_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++) target->data[i] = mat1->data[i] * mat2->data[i] * (1-mat2->data[i]);

  return 0;
}

int apply_tanh_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++) target->data[i] = 0.5 * mat1->data[i] * (1 + mat2->data[i]) * (1 - mat2->data[i]);
  return 0;
}

int apply_rectified_linear_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++) target->data[i] =  mat1->data[i] * (mat2->data[i] > 0 ? 1 : 0);
  return 0;
}

int apply_rectified_linear_smooth_deriv(eigenmat* mat1, eigenmat* mat2, eigenmat* target) {
  int len = mat1->size[0]*mat1->size[1];

  if (mat1->is_trans != mat2->is_trans)
    return ERROR_TRANSPOSEDNESS;

  if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
    mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;

  #pragma omp parallel for
  for (int i = 0; i < len; i++) target->data[i] =  mat1->data[i] * (1 - exp(-mat2->data[i]));
  return 0;
}

int write_at(eigenmat* mat, int row, int col, float val)
{
    if (row >= mat->size[0] || col >= mat->size[1] || row < 0 || col < 0)
      return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->data[col * mat->size[0] + row] = val;

    return 0;
}

float read_from(eigenmat* mat, int row, int col, int* err_code)
{
    *err_code = 0;
    if (row >= mat->size[0] || col >= mat->size[1] || row < 0 || col < 0)
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;

    return mat->data[col * mat->size[0] + row];
}

int assign_scalar(eigenmat* mat, float alpha) {
  int len = mat->size[0]*mat->size[1];
  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  eig_mat.setConstant(alpha);
  return 0;
}

int mult_by_scalar(eigenmat* mat, float alpha, eigenmat* target) {
  const int len = mat->size[0]*mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat * alpha;
  return 0;
}

int divide_by_scalar(eigenmat* mat, float alpha, eigenmat* target) {
  const int len = mat->size[0]*mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat / alpha;
  return 0;
}

int add_scalar(eigenmat* mat, float alpha, eigenmat* target) {
  const int len = mat->size[0] * mat->size[1];
  if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
    return ERROR_INCOMPATIBLE_DIMENSIONS;
  Eigen::Map<Eigen::ArrayXf> eig_mat(mat->data, len);
  Eigen::Map<Eigen::ArrayXf> eig_target(target->data, len);
  eig_target = eig_mat + alpha;
  return 0;
}

float euclid_norm(eigenmat* mat) {
  const int len = mat->size[0]*mat->size[1];
  Eigen::Map<Eigen::VectorXf> eig_mat(mat->data, len);
  return eig_mat.norm();
}

int selectCols(eigenmat* source, eigenmat* target, eigenmat* indices){
  const int n = indices->size[1] * indices->size[0];
  const int h = source->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    target_offset = i * h;
    source_offset = (int)indices->data[i] * h;
    for (int j = 0; j < h; j++) {
      target->data[target_offset + j] = source->data[source_offset + j];
    }
  }
  return 0;
}

int selectRows(eigenmat* source, eigenmat* target, eigenmat* indices){
  const int n = indices->size[1] * indices->size[0];
  const int w = source->size[1], h_source = source->size[0], h_target = target->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int j = 0; j < w; j++) {
    for (int i = 0; i < n; i++) {
      target->data[j * h_target + i] = source->data[j * h_source + (int)indices->data[i]];
    }
  }
  return 0;
}

int swapColumns(eigenmat* source, eigenmat* target, eigenmat* indices1, eigenmat* indices2){
  const int n = indices1->size[1] * indices1->size[0];
  const int h = source->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    source_offset = (int)indices1->data[i] * h;
    target_offset = (int)indices2->data[i] * h;
    for (int j = 0; j < h; j++) {
      float temp = target->data[target_offset + j];
      target->data[target_offset + j] = source->data[source_offset + j];
      source->data[source_offset + j] = temp;
    }
  }
  return 0;
}

int swapRows(eigenmat* source, eigenmat* target, eigenmat* indices1, eigenmat* indices2){
  const int n = indices1->size[1] * indices1->size[0];
  const int w = source->size[1], h_source = source->size[0], h_target = target->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int j = 0; j < w; j++) {
    for (int i = 0; i < n; i++) {
      int source_index = j * h_source + (int)indices1->data[i];
      int target_index = j * h_target + (int)indices2->data[i];
      float temp = target->data[target_index];
      target->data[target_index] = source->data[source_index];
      source->data[source_index] = temp;
    }
  }
  return 0;
}

int setSelectedCols(eigenmat* source, eigenmat* target, eigenmat* indices){
  const int n = indices->size[1] * indices->size[0];
  const int h = source->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    source_offset = i * h;
    target_offset = (int)indices->data[i] * h;
    for (int j = 0; j < h; j++) {
      target->data[target_offset + j] = source->data[source_offset + j];
    }
  }
  return 0;
}

int setSelectedRows(eigenmat* source, eigenmat* target, eigenmat* indices){
  const int n = indices->size[1] * indices->size[0];
  const int w = source->size[1], h_source = source->size[0], h_target = target->size[0];

  int target_offset, source_offset;
  #pragma omp parallel for
  for (int j = 0; j < w; j++) {
    for (int i = 0; i < n; i++) {
      target->data[j * h_target + (int)indices->data[i]] = source->data[j * h_source + i];
    }
  }
  return 0;
}

int extract_patches(eigenmat* images, eigenmat* patches, eigenmat* width_offset,
                    eigenmat* height_offset, eigenmat* flip, int img_width,
                    int img_height, int patch_width, int patch_height)
{
    int num_images = images->size[1];
    int num_colors = images->size[0] / (img_width * img_height);

    if (patches->size[1]  != num_colors * patch_width * patch_height || patches->size[0] != num_images)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (width_offset->size[0] * width_offset->size[1] != num_images)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (height_offset->size[0] * height_offset->size[1] != num_images)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (flip->size[0] * flip->size[1] != num_images)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    for (int image_id=0; image_id<num_images; ++image_id)
    {
        for (int dest_col=0; dest_col<patch_width; ++dest_col)
        {
            int source_col = int(width_offset->data[image_id]) + dest_col;
            if (flip->data[image_id] > 0.5)
            {
                source_col = (img_width - source_col - 1);
            }

            for (int dest_row=0; dest_row<patch_height; ++dest_row)
            {
                int source_row = int(height_offset->data[image_id]) + dest_row;
                for (int color=0; color<num_colors; ++color)
                {
                    unsigned long dest_index = image_id + num_images * (dest_col  + patch_width * (dest_row + patch_height * color));
                    unsigned long source_index = source_col + img_width * (source_row + img_height * (color + num_colors * image_id));

                    patches->data[dest_index] = images->data[source_index];
                }
            }
        }
    }

    return 0;
}

int blockify(eigenmat* source, eigenmat* target, int blocksize) {
  const int w = source->size[1], h = source->size[0];
  #pragma omp parallel for
  for (int i = 0; i < w; i++) {
    const int off = i * h;
    for (int j = 0; j < h; j++) {
      target->data[off + j] = source->data[off + (j / blocksize) * blocksize];
    }
  }
  return 0;
}
