#include <catch.hpp>
#include <iostream>
#include "gsl/gsl_matrix.h"
#include "param.h"
#include <algorithm>
#include <limits>
#include <numeric>

using namespace std;

TEST_CASE( "LMM functions2", "[lmm2]" ) {

  size_t n_cvt = 2;
  size_t e_mode = 0;

  REQUIRE(GetabIndex(4, 9, 2)  == 0 );
  REQUIRE(GetabIndex(4, 9, 16) == 56);
}
