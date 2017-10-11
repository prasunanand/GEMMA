#include <catch.hpp>
#include <iostream>
#include "gsl/gsl_blas.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_permutation.h"
#include "gsl/gsl_vector.h"
#include "param.h"
#include "lapack.h"
#include "lmm.h"
#include <algorithm>
#include <limits>
#include <numeric>

using namespace std;

TEST_CASE( "LMM functions", "[lmm]" ) {

  size_t n_cvt = 3;
  size_t e_mode = 0;

  REQUIRE(GetabIndex(4, 9, 2)  == 0 );
  REQUIRE(GetabIndex(4, 9, 16) == 56);

  double G_data[] = { 212,   7, 11, 12, 30,
                      11,  101, 34,  1, -7,
                      151,-101, 96,  1, 73,
                      87,  102, 64, 19, 67,
                     -21,  10, 334, 22, -2
                    };

  char func_name = 'R';
  double l_min = 0.00001;
  double l_max = 10;
  size_t n_region = 10;
  size_t ni_test = 5;//UtW.size1;
  size_t n_ph = 1;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  gsl_matrix *G = gsl_matrix_alloc(5,5);
  copy(G_data, G_data+25, G->data);

  gsl_matrix* W = gsl_matrix_alloc(5,1);
  gsl_matrix_set_all(W, 1.0);

  double y_data[] = {3, 14 ,-5, 18, 6};
  gsl_vector* y = gsl_vector_alloc(5);
  copy(y_data, y_data+5, y->data);

  gsl_matrix *Y = gsl_matrix_alloc(ni_test, n_ph);
  gsl_matrix *U = gsl_matrix_alloc(Y->size1, Y->size1);
  gsl_matrix *UtW = gsl_matrix_alloc(Y->size1, W->size2);
  gsl_matrix *UtY = gsl_matrix_alloc(Y->size1, Y->size2);
  gsl_vector *Uty = gsl_vector_alloc(U->size2);
  gsl_vector *eval = gsl_vector_calloc(Y->size1);
  gsl_vector *ab = gsl_vector_alloc(n_index);

  gsl_blas_dgemv(CblasTrans, 1.0, U, y, 0.0, Uty);

  double trace_G = EigenDecomp_Zeroed(G, U, eval, 0);
  REQUIRE( abs(trace_G - 85.2) < 0.001);
  
  gsl_matrix* Uab = gsl_matrix_alloc(ni_test, n_index);

  CalcUab(UtW, Uty, Uab);
  // gsl_matrix_set_zero(Uab);
  // gsl_matrix_set_all(Uab, 1.0);
  gsl_vector_set_all(ab, 100.0);

  cout << "ni_test is " << ni_test << endl;
  cout << "n_cvt is " << n_cvt << endl;

  Calcab(W, y, ab);
  FUNC_PARAM param0 = {true, ni_test, n_cvt, eval, Uab, ab, 0};

  double l = 6;
  double beta;
  double se;
  double p_wald;
  double lambda;
  double logl_H0;

  gsl_vector_view UtY_col = gsl_matrix_column(UtY, 0);

  double dev1_l = LogL_dev1(10, &param0);
  cout << "dev1_l is " << dev1_l << endl; 
  double dev2_l = LogL_dev2(0, &param0);
  cout << "dev2_l is " << dev2_l << endl; 
  double dev_f =  LogRL_f(0, &param0);
  cout << "dev_f is " << dev_f << endl; 
  double dev1_r = LogRL_dev1(0, &param0);
  cout << "dev1_r is " << dev1_r << endl; 
  double dev2_r = LogRL_dev2(0, &param0);
  cout << "dev2_r is " << dev2_r << endl; 
  LogL_dev12(0, &param0, &dev1_l, &dev2_l);
  LogRL_dev12(0, &param0, &dev1_r, &dev2_r);


  // CalcLambda(func_name, eval, UtW, &UtY_col.vector, l_min, l_max, n_region, lambda, logl_H0);

  double pve, pve_se;
  // CalcPve(eval,  UtW, Uty, lambda, trace_G, pve,  pve_se);
}
