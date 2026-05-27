#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ivf.hpp"

#include <stdexcept>
#include <vector>

TEST_CASE("distance_hamming: identical vectors have distance 0")
{
    genivf::Point p1(1, { 0xAB, 0xCD });
    genivf::Point p2(2, { 0xAB, 0xCD });
    CHECK_EQ(genivf::distance_hamming(p1, p2), 0u);
}

TEST_CASE("distance_hamming: complement vectors have maximum distance")
{
    // 0xAA = 10101010, 0x55 = 01010101 — every bit in every byte differs.
    genivf::Point p1(1, { 0xAA, 0x55 });
    genivf::Point p2(2, { 0x55, 0xAA });
    CHECK_EQ(genivf::distance_hamming(p1, p2), 16u); // 8 bits × 2 bytes
}

TEST_CASE("distance_hamming: known single-bit difference")
{
    // 0x01 = 00000001, 0x03 = 00000011 — only bit 1 differs.
    genivf::Point p1(1, { 0x01 });
    genivf::Point p2(2, { 0x03 });
    CHECK_EQ(genivf::distance_hamming(p1, p2), 1u);
}

TEST_CASE("distance_jaccard: identical vectors have distance 0")
{
    genivf::Point p1(1, { 0xFF });
    genivf::Point p2(2, { 0xFF });
    CHECK_EQ(genivf::distance_jaccard(p1, p2), doctest::Approx(0.0f));
}

TEST_CASE("distance_jaccard: fully disjoint vectors have distance 1")
{
    // 0xAA = 10101010, 0x55 = 01010101 — no bits in common.
    genivf::Point p1(1, { 0xAA });
    genivf::Point p2(2, { 0x55 });
    CHECK_EQ(genivf::distance_jaccard(p1, p2), doctest::Approx(1.0f));
}

TEST_CASE("distance_jaccard: known partial overlap")
{
    // 0xF0 = 11110000, 0xCC = 11001100
    // AND  = 11000000  →  popcount = 2  (intersection)
    // OR   = 11111100  →  popcount = 6  (union)
    // Jaccard distance = 1 - 2/6 = 2/3
    genivf::Point p1(1, { 0xF0 });
    genivf::Point p2(2, { 0xCC });
    CHECK_EQ(genivf::distance_jaccard(p1, p2), doctest::Approx(2.0f / 3.0f));
}

// ─── IndexIVF construction
// ────────────────────────────────────────────────────

TEST_CASE("IndexIVF: valid construction leaves index untrained")
{
    genivf::IndexIVF ivf(3, 4);
    CHECK_FALSE(ivf.is_trained());
}

TEST_CASE("IndexIVF: throws on dim == 0")
{
    CHECK_THROWS_AS(genivf::IndexIVF(3, 0), std::invalid_argument);
}

TEST_CASE("IndexIVF: throws on num_cells == 0")
{
    CHECK_THROWS_AS(genivf::IndexIVF(0, 4), std::invalid_argument);
}

// ─── train ───────────────────────────────────────────────────────────────────

TEST_CASE("IndexIVF::train: marks index as trained")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(points);
    CHECK(ivf.is_trained());
}

TEST_CASE("IndexIVF::train: throws when fewer points than cells")
{
    genivf::IndexIVF ivf(4, 1);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
    };
    CHECK_THROWS_AS(ivf.train(points), std::invalid_argument);
}

TEST_CASE("IndexIVF::train: throws on dimension mismatch")
{
    genivf::IndexIVF ivf(2, 2); // expects 2 bytes per point
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } }, // only 1 byte — wrong
        { 2, { 0xFF } },
        { 3, { 0x0F } },
    };
    CHECK_THROWS_AS(ivf.train(points), std::invalid_argument);
}

TEST_CASE("IndexIVF::train: re-training is safe (clears old state)")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(points);
    ivf.train(points); // must not throw or corrupt
    CHECK(ivf.is_trained());
}

// ─── add ─────────────────────────────────────────────────────────────────────

TEST_CASE("IndexIVF::add: throws if not trained")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> points = { { 1, { 0xAA } } };
    CHECK_THROWS_AS(ivf.add(points), std::logic_error);
}

TEST_CASE("IndexIVF::add: throws on duplicate id")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> train_pts = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(train_pts);
    ivf.add(train_pts);

    // Try to add a point whose id already exists.
    std::vector<genivf::Point> dup = { { 1, { 0x55 } } };
    CHECK_THROWS_AS(ivf.add(dup), std::invalid_argument);
}

TEST_CASE("IndexIVF::add: throws on dimension mismatch")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> train_pts = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(train_pts);

    std::vector<genivf::Point> bad = { { 5,
                                         { 0x01, 0x02 } } }; // 2 bytes, dim=1
    CHECK_THROWS_AS(ivf.add(bad), std::invalid_argument);
}

// ─── search ──────────────────────────────────────────────────────────────────

TEST_CASE("IndexIVF::search: throws if not trained")
{
    genivf::IndexIVF ivf(2, 1);
    genivf::Point query(0, { 0x01 });
    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)ivf.search(query, 1), std::logic_error);
}

TEST_CASE("IndexIVF::search: throws on bad nprobe")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> pts = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(pts);
    ivf.add(pts);
    genivf::Point query(0, { 0x01 });
    // HACK: void cast to suppress unused-result warning
    CHECK_THROWS_AS((void)ivf.search(query, 1, 0),
                    std::invalid_argument); // nprobe=0
    CHECK_THROWS_AS((void)ivf.search(query, 1, 9),
                    std::invalid_argument); // nprobe > num_cells
}

TEST_CASE("IndexIVF::search: k==0 returns empty result")
{
    genivf::IndexIVF ivf(2, 1);
    std::vector<genivf::Point> pts = {
        { 1, { 0x00 } },
        { 2, { 0xFF } },
        { 3, { 0x0F } },
        { 4, { 0xF0 } },
    };
    ivf.train(pts);
    ivf.add(pts);
    genivf::Point query(0, { 0x01 });
    auto results = ivf.search(query, 0, 2);
    CHECK(results.empty());
}

TEST_CASE("IndexIVF::search: returns at most k results")
{
    genivf::IndexIVF ivf(2, 1, 0);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    ivf.train(points);
    ivf.add(points);
    genivf::Point query(0, { 0x00 });
    auto results = ivf.search(query, 2, 2);
    CHECK_LE(results.size(), static_cast<size_t>(2));
}

TEST_CASE("IndexIVF::search: nearest point has distance 0 (exhaustive probe)")
{
    // Use nprobe == num_cells so the search is exhaustive and the result is
    // independent of how k-means happens to partition the data.
    genivf::IndexIVF ivf(2, 1, 0);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    ivf.train(points);
    ivf.add(points);

    genivf::Point query(0, { 0x00 });
    auto results = ivf.search(query, 1, 2, genivf::MetricType::HAMMING);
    REQUIRE_FALSE(results.empty());
    // The query is identical to point 1, so distance must be 0.
    CHECK_EQ(results[0].id, static_cast<size_t>(1));
    CHECK_EQ(results[0].distance, doctest::Approx(0.0));
}

TEST_CASE("IndexIVF::search: results are ordered by ascending distance")
{
    genivf::IndexIVF ivf(2, 1, 42);
    std::vector<genivf::Point> points = {
        { 1, { 0x00 } },
        { 2, { 0x01 } },
        { 3, { 0xFE } },
        { 4, { 0xFF } },
    };
    ivf.train(points);
    ivf.add(points);

    genivf::Point query(0, { 0x00 });
    auto results = ivf.search(query, 4, 2, genivf::MetricType::HAMMING);
    for (size_t i = 1; i < results.size(); ++i) {
        CHECK_LE(results[i - 1].distance, results[i].distance);
    }
}
