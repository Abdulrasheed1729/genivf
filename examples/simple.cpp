#include "io.hpp"
#include "ivf.hpp"

#include <print>
#include <vector>

int
main()
{
    // Build a small binary IVF over 1-byte (8-bit) vectors.
    // 2 Voronoi cells, 1 byte per vector, fixed seed for reproducibility.
    genivf::IndexIVF ivf(2, 2, 42);

    std::vector<genivf::Point> points = {
        { 1, { 0x00, 0x02 } },  // 00000000 00000010
        { 2, { 0x01, 0x03 } },  // 00000001 00000011
        { 3, { 0x03, 0x04 } },  // 00000011 00000100
        { 4, { 0xF0, 0xF1 } },  // 11110000 11110001
        { 5, { 0xFE, 0xAD } },  // 11111110 10101101
        { 6, { 0xFF, 0xF0 } },  // 11111111 11110000
        { 7, { 0x12, 0x34 } },  // 00010010 00110100
        { 8, { 0x56, 0x78 } },  // 01010110 01111000
        { 9, { 0x9A, 0xBC } },  // 10011010 10111100
        { 10, { 0xDE, 0xF0 } }, // 11011110 11110000
        { 11, { 0x11, 0x22 } }, // 00010001 00100010
        { 12, { 0x33, 0x44 } }, // 00110011 01000100
    };

    ivf.train(points, 100, 1e-6);
    ivf.add(points);
    genivf::io::save_index(ivf, "hello.givf");

    genivf::Point query(0, { 0x01, 0x03 }); // 00000001 00000011

    auto ivfl = genivf::io::load_index("hello.givf");

    std::println("============== Hamming ====================");
    std::println("Top-3 nearest to 0x01 (Hamming distance):");
    for (const auto& r :
         ivfl.search(query, 3, 2, genivf::MetricType::HAMMING)) {
        std::println("  id={:2d}  distance={:.0f}", r.id, r.distance);
    }

    std::println("============== Jaccard ====================");
    std::println("\nTop-3 nearest to 0x01 (Jaccard distance):");
    for (const auto& r :
         ivfl.search(query, 3, 2, genivf::MetricType::JACCARD)) {
        std::println("  id={:2d}  distance={:.4f}", r.id, r.distance);
    }

    std::println("============== L2 ====================");
    std::println("\nTop-3 nearest to 0x01 (L2 distance):");
    for (const auto& r : ivfl.search(query, 3, 2, genivf::MetricType::L2)) {
        std::println("  id={:2d}  distance={:.4f}", r.id, r.distance);
    }

    return 0;
}
