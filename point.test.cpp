#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "genivf.hpp"

TEST_CASE("instance with vectors and id")
{
    genivf::Point p( std::vector<float>{ 1.0, 2.0 }, 2 );
    CHECK_EQ(p.values, std::vector<float>{1.0, 2.0});
    CHECK_EQ(p.id, 2);
}
