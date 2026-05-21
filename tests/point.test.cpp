#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "types.hpp"

#include <stdexcept>
#include <vector>

TEST_CASE("Point: construct with vector and id")
{
    genivf::Point p(2, std::vector<uint8_t>{ 0x01, 0x02 });
    CHECK_EQ(p.id, static_cast<size_t>(2));
    CHECK_EQ(p.values, std::vector<uint8_t>{ 0x01, 0x02 });
}

TEST_CASE("Point: construct with initialiser list")
{
    genivf::Point p(2, { 0x01, 0x02 });
    CHECK_EQ(p.id, static_cast<size_t>(2));
    CHECK_EQ(p.values, std::vector<uint8_t>{ 0x01, 0x02 });
}

TEST_CASE("Point: operator[] returns correct byte")
{
    genivf::Point p(1, { 0xAA, 0x55 });
    CHECK_EQ(p[0], static_cast<uint8_t>(0xAA));
    CHECK_EQ(p[1], static_cast<uint8_t>(0x55));
}

TEST_CASE("Point: at() throws std::out_of_range for out-of-bounds index")
{
    genivf::Point p(1, { 0x01, 0x02 });
    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)p.at(2), std::out_of_range);
}
