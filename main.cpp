#include "genivf.hpp"
#include "io.hpp"

#include <print>
#include <vector>

int
main()
{
    // Build a small binary IVF over 1-byte (8-bit) vectors.
    // 2 Voronoi cells, 1 byte per vector, fixed seed for reproducibility.
    genivf::IndexIVF ivf(2, 2, 42);

    std::vector<genivf::Point> points = {
        { 1, { 0x00, 0x02 } }, // 00000000
        { 2, { 0x01, 0x03 } }, // 00000001
        { 3, { 0x03, 0x04 } }, // 00000011
        { 4, { 0xF0, 0xF1 } }, // 11110000
        { 5, { 0xFE, 0xAD } }, // 11111110
        { 6, { 0xFF, 0xF0 } }, // 11111111
    };

    ivf.train(points);
    ivf.add(points);

    genivf::Point query(0, { 0x01, 0x03 }); // 00000001

    auto ivfl = genivf::io::load_index("hello.givf");

    std::println("Top-3 nearest to 0x01 (Hamming distance):");
    for (const auto& r :
         ivfl.search(query, 3, 2, genivf::MetricType::HAMMING)) {
        std::println("  id={:2d}  distance={:.0f}", r.id, r.distance);
    }

    std::println("\nTop-3 nearest to 0x01 (Jaccard distance):");
    for (const auto& r :
         ivfl.search(query, 3, 2, genivf::MetricType::JACCARD)) {
        std::println("  id={:2d}  distance={:.4f}", r.id, r.distance);
    }

    return 0;
}
