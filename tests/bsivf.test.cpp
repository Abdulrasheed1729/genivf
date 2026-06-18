#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "bsivf.hpp"
#include "doctest.h"
#include "types.hpp"
#include <vector>
#include <iostream>

TEST_CASE("IndexBSIVF: basic construction and is_trained")
{
    genivf::IndexBSIVF index(1, 10);
    CHECK_FALSE(index.is_trained());
    index.construct_centroids(3);
    CHECK(index.is_trained());
}

TEST_CASE("IndexBSIVF: duplicate points in search results")
{
    // Initialize with 10 points
    genivf::IndexBSIVF index(1, 10);
    std::vector<genivf::Point> points;
    for (size_t i = 0; i < 10; ++i) {
        points.push_back({ i, { static_cast<uint8_t>(i * 10) } });
    }
    index.construct_centroids(3);
    index.add(points);

    genivf::Point query(99, { 10 });
    auto results = index.search(query, 5, 4, 1, genivf::MetricType::HAMMING);

    std::cout << "Search results size: " << results.size() << std::endl;
    for (const auto& r : results) {
        std::cout << "ID: " << r.id << ", Distance: " << r.distance << std::endl;
    }

    // Verify there are no duplicate IDs in the results
    std::vector<size_t> ids;
    for (const auto& r : results) {
        ids.push_back(r.id);
    }
    std::sort(ids.begin(), ids.end());
    auto unique_end = std::unique(ids.begin(), ids.end());
    CHECK_EQ(ids.end(), unique_end); // No duplicates!
}

TEST_CASE("IndexBSIVF: out of bounds bug when adding fewer points than ntotal")
{
    // Initialize with 100 points, but only add 25
    genivf::IndexBSIVF index(1, 100);
    index.construct_centroids(25); // centroids at 0, 25, 50, 75

    std::vector<genivf::Point> points;
    for (size_t i = 0; i < 25; ++i) {
        points.push_back({ i, { static_cast<uint8_t>(i) } });
    }
    index.add(points);

    // This should not crash / access out of bounds
    genivf::Point query(99, { 0 });
    auto results = index.search(query, 2, 25, 4, genivf::MetricType::HAMMING);
    CHECK(results.size() > 0);
}
