#include <stdexcept>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "genivf.hpp"

TEST_CASE("instance with vectors and id")
{
    genivf::Point p(std::vector<float>{ 1.0, 2.0 }, 2);
    CHECK_EQ(p.values, std::vector<float>{ 1.0, 2.0 });
    CHECK_EQ(p.id, static_cast<size_t>(2));
}

TEST_CASE("instance with initialiser list")
{
    genivf::Point p(2, { 1.0, 2.0 });
    CHECK_EQ(p.values, std::vector<float>{ 1.0, 2.0 });
    CHECK_EQ(p.id, static_cast<size_t>(2));
}

TEST_CASE("test indexing")
{
    genivf::Point p(std::vector<float>{ 1.0, 2.0 }, 2);
    CHECK_EQ(p[0], 1.0);
    CHECK_EQ(p[1], 2.0);
}

TEST_CASE("accessing an out of bound should throw")
{
    genivf::Point p(std::vector<float>{ 1.0, 2.0 }, 2);
    CHECK_THROWS_AS(p[2], std::out_of_range);
}
