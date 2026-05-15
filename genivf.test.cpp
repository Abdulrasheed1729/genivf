#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "genivf.hpp"
#include <stdexcept>
#include <cmath>

TEST_CASE("IndexIVF instance: should work with the right params") {
    size_t num_cells = 3;
    size_t dim = 3;
    genivf::IndexIVF ivf (num_cells, dim);
}

TEST_CASE("IndexIVF instance: should throw with `dim = 0` or `num_cells <= 0`") {
    size_t num_cells = 0;
    size_t dim = 0;
    CHECK_THROWS_AS(genivf::IndexIVF ivf (num_cells, dim), std::invalid_argument);

    num_cells = -1;
    CHECK_THROWS_AS(genivf::IndexIVF ivf (num_cells, dim), std::invalid_argument);
}

TEST_CASE("IndexIVF instance: new instance should be untrained") {
    size_t num_cells = 3;
    size_t dim = 3;
    genivf::IndexIVF ivf (num_cells, dim);
    CHECK_EQ(ivf.is_trained(), false);

}

TEST_CASE("test the distance l2 and l2 sq.") {
    genivf::Point p1 = {1, {1.0, 2.0}};
    genivf::Point p2 = {2, {3.0, 4.0}};
    CHECK_EQ(genivf::IndexIVF::distance_l2(p1, p2),
                                    2.0*std::sqrt(2.0));
    CHECK_EQ(genivf::IndexIVF::distance_l2_sq(p1, p2), 8.0);
}

TEST_CASE("test out the distance l2") {
    std::vector<genivf::Point> points = {
        {1,{1.0, 2.0, 3.0}},
        {2,{4.0, 5.0, 6.0}},
        {3,{7.0, 8.0, 9.0}},
        {4,{10.0, 11.0, 12.0}},
        {5,{13.0, 14.0, 15.0}},
        {6,{16.0, 17.0, 18.0}},
    };
}

// - [x] Test out the distance functions
//      - [x] bothe the l2 and the l2 squared
// - [ ] Test out the train functions
// - [ ] Test out add 
// NOTE: train must be called before add
// and there should be some kind of exception throw while calling add before train
