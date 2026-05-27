#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "flat.hpp"
#include "doctest.h"
#include "io.hpp"

#include <filesystem>
#include <stdexcept>
#include <vector>

TEST_CASE("IndexFlat: basic add and search")
{
    genivf::IndexFlat index(1, 4);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    index.add(points);

    genivf::Point query(0, { 0x00 });
    auto results = index.search(query, 2, 1, genivf::MetricType::HAMMING);

    REQUIRE_EQ(results.size(), 2u);
    CHECK_EQ(results[0].id, 1u);
    CHECK_EQ(results[0].distance, 0.0);
}

TEST_CASE("IndexFlat: k == 0 returns empty")
{
    genivf::IndexFlat index(1, 1);
    std::vector<genivf::Point> points = { { 1, { 0xAA } } };
    index.add(points);
    genivf::Point query(0, { 0xAA });
    auto results = index.search(query, 0);
    CHECK(results.empty());
}

TEST_CASE("IndexFlat: throws on dimension mismatch")
{
    genivf::IndexFlat index(2, 1);
    std::vector<genivf::Point> bad = { { 1, { 0x01 } } };
    CHECK_THROWS_AS(index.add(bad), std::invalid_argument);
}

TEST_CASE("IndexFlat: throws on duplicate id")
{
    genivf::IndexFlat index(1, 2);
    std::vector<genivf::Point> pts = { { 1, { 0x00 } }, { 2, { 0xFF } } };
    index.add(pts);
    std::vector<genivf::Point> dup = { { 1, { 0x55 } } };
    CHECK_THROWS_AS(index.add(dup), std::invalid_argument);
}

TEST_CASE("IndexFlat: results ordered by ascending distance")
{
    genivf::IndexFlat index(1, 4);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0x10 } },
        { 4, { 0xFF } },
    };
    index.add(points);

    genivf::Point query(0, { 0x00 });
    auto results = index.search(query, 4, 1, genivf::MetricType::HAMMING);
    for (size_t i = 1; i < results.size(); ++i) {
        CHECK_LE(results[i - 1].distance, results[i].distance);
    }
}

TEST_CASE("IndexFlat: L2 distance search")
{
    genivf::IndexFlat index(1, 3);
    std::vector<genivf::Point> points = {
        { 1, { 0x01 } },
        { 2, { 0x02 } },
        { 3, { 0x04 } },
    };
    index.add(points);

    genivf::Point query(0, { 0x01 });
    auto results = index.search(query, 3, 1, genivf::MetricType::L2);
    REQUIRE_EQ(results.size(), 3u);
    CHECK_EQ(results[0].id, 1u);
    CHECK_EQ(results[0].distance, 0.0);
}

TEST_CASE("IndexFlat: Jaccard distance search")
{
    genivf::IndexFlat index(1, 2);
    std::vector<genivf::Point> points = {
        { 1, { 0xFF } },
        { 2, { 0x00 } },
    };
    index.add(points);

    genivf::Point query(0, { 0xFF });
    auto results = index.search(query, 2, 1, genivf::MetricType::JACCARD);
    REQUIRE_EQ(results.size(), 2u);
    CHECK_EQ(results[0].id, 1u);
    CHECK_EQ(results[0].distance, 0.0);
}

// ─── Flat index IO round-trip ────────────────────────────────────────────

TEST_CASE("IndexFlat: save/load round-trip")
{
    genivf::IndexFlat original(1, 4);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    original.add(points);

    auto tmp = std::filesystem::temp_directory_path() / "flat_rt.givf";
    genivf::io::save_flat_index(original, tmp);

    auto loaded = genivf::io::load_flat_index(tmp);

    genivf::Point query(0, { 0x00 });
    auto before = original.search(query, 4, 1, genivf::MetricType::HAMMING);
    auto after = loaded.search(query, 4, 1, genivf::MetricType::HAMMING);

    REQUIRE_EQ(before.size(), after.size());
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK_EQ(before[i].id, after[i].id);
        CHECK_EQ(before[i].distance, doctest::Approx(after[i].distance));
    }

    std::filesystem::remove(tmp);
}
