#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "utils.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

TEST_CASE("distance_l2_sq: identical vectors have distance 0")
{
    uint8_t a[] = { 1, 2, 3, 4 };
    uint8_t b[] = { 1, 2, 3, 4 };
    CHECK_EQ(genivf::distance_l2_sq(a, b, 4), 0.0);
}

TEST_CASE("distance_l2_sq: known squared difference")
{
    uint8_t a[] = { 0, 0 };
    uint8_t b[] = { 3, 4 };
    CHECK_EQ(genivf::distance_l2_sq(a, b, 2), 25.0); // 3^2 + 4^2
}

TEST_CASE("distance_l2: known distance")
{
    uint8_t a[] = { 0, 0 };
    uint8_t b[] = { 3, 4 };
    CHECK_EQ(genivf::distance_l2(a, b, 2), 5.0);
}

TEST_CASE("distance_l2: zero for identical")
{
    uint8_t a[] = { 0xAB, 0xCD };
    uint8_t b[] = { 0xAB, 0xCD };
    CHECK_EQ(genivf::distance_l2(a, b, 2), 0.0);
}

TEST_CASE("distance_hamming: 64-bit word alignment edge cases")
{
    uint8_t a[9] = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 };
    uint8_t b[9] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF };
    // First 8 bytes: all bits differ → 64. Last byte: 8 bits differ → 8. Total = 72.
    CHECK_EQ(genivf::distance_hamming(a, b, 9), 72u);
}

TEST_CASE("distance_hamming: non-multiple-of-8 length")
{
    uint8_t a[] = { 0xFF, 0x00, 0xF0 };
    uint8_t b[] = { 0x00, 0xFF, 0x0F };
    // Byte 0: popcount(0xFF ^ 0x00) = 8
    // Byte 1: popcount(0x00 ^ 0xFF) = 8
    // Byte 2: popcount(0xF0 ^ 0x0F) = 8
    CHECK_EQ(genivf::distance_hamming(a, b, 3), 24u);
}

TEST_CASE("distance_jaccard: identical distance 0")
{
    uint8_t a[] = { 0xFF, 0xFF };
    uint8_t b[] = { 0xFF, 0xFF };
    CHECK_EQ(genivf::distance_jaccard(a, b, 2), doctest::Approx(0.0f));
}

TEST_CASE("distance_jaccard: disjoint distance 1")
{
    uint8_t a[] = { 0xAA };
    uint8_t b[] = { 0x55 };
    CHECK_EQ(genivf::distance_jaccard(a, b, 1), doctest::Approx(1.0f));
}

// ─── K-mer utilities ─────────────────────────────────────────────────────

TEST_CASE("base_to_bits: valid bases")
{
    CHECK_EQ(genivf::base_to_bits('A'), 0);
    CHECK_EQ(genivf::base_to_bits('C'), 1);
    CHECK_EQ(genivf::base_to_bits('G'), 2);
    CHECK_EQ(genivf::base_to_bits('T'), 3);
    CHECK_EQ(genivf::base_to_bits('a'), 0);
    CHECK_EQ(genivf::base_to_bits('c'), 1);
    CHECK_EQ(genivf::base_to_bits('g'), 2);
    CHECK_EQ(genivf::base_to_bits('t'), 3);
}

TEST_CASE("base_to_bits: invalid returns -1")
{
    CHECK_EQ(genivf::base_to_bits('N'), -1);
    CHECK_EQ(genivf::base_to_bits('X'), -1);
    CHECK_EQ(genivf::base_to_bits(' '), -1);
}

TEST_CASE("kmer_vector_size: known values")
{
    CHECK_EQ((genivf::kmer_vector_size<1>()), 4u);
    CHECK_EQ((genivf::kmer_vector_size<2>()), 16u);
    CHECK_EQ((genivf::kmer_vector_size<3>()), 64u);
    CHECK_EQ((genivf::kmer_vector_size<4>()), 256u);
    CHECK_EQ((genivf::kmer_vector_size<5>()), 1024u);
}

TEST_CASE("compute_binary_kmer_vector: exact k-mer length sets one bit")
{
    constexpr size_t K = 2;
    constexpr size_t DIM = (size_t(1) << (2 * K));
    std::array<uint8_t, DIM> out{};

    // "AC" → codes 0,1 → val = (0 << 2 | 1) = 1
    genivf::compute_binary_kmer_vector<K, DIM>("AC", 2, out);
    CHECK_EQ(out[1], 1u);
    // No other bits should be set
    for (size_t i = 0; i < DIM; ++i) {
        if (i != 1) CHECK_EQ(out[i], 0u);
    }
}

TEST_CASE("compute_binary_kmer_vector: sliding window sets multiple bits")
{
    constexpr size_t K = 2;
    constexpr size_t DIM = (size_t(1) << (2 * K));
    std::array<uint8_t, DIM> out{};

    // "ACGT" → codes: A=0, C=1, G=2, T=3
    // k-mers: AC(1), CG(6=4+2), GT(11=8+3=0xB & 0xF = 11=0xB)
    genivf::compute_binary_kmer_vector<K, DIM>("ACGT", 4, out);
    CHECK_EQ(out[1], 1u);  // AC = (0<<2 | 1) = 1
    CHECK_EQ(out[6], 1u);  // CG = (1<<2 | 2) = 6
    CHECK_EQ(out[11], 1u); // GT = (2<<2 | 3) = 11
    CHECK_EQ(out[0], 0u);
}

TEST_CASE("compute_binary_kmer_vector: skips N characters")
{
    constexpr size_t K = 2;
    constexpr size_t DIM = (size_t(1) << (2 * K));
    std::array<uint8_t, DIM> out{};

    // "ACNGT" → N resets the rolling hash, so "NG" and "GT" are valid after
    genivf::compute_binary_kmer_vector<K, DIM>("ACNGT", 5, out);
    // AC = 1 set
    CHECK_EQ(out[1], 1u);
    // After N reset: NG = 2-bit+N invalid → N is -1, so val=0, run=0
    // Then G=2 → val = (0<<2 | 2) = 2, run=1, continue
    // Then T=3 → val = (2<<2 | 3) = 11, run=2 → set out[11]
    CHECK_EQ(out[11], 1u);
}

TEST_CASE("pack_kmer_vector_endian: round-trip consistency")
{
    constexpr size_t DIM = 16;
    std::array<uint8_t, DIM> vec{};
    vec[0] = 1;
    vec[7] = 1;
    vec[15] = 1;

    uint8_t packed[2] = {};
    genivf::pack_kmer_vector_endian<DIM>(vec, packed);

    // Bit 0 → byte 0, bit 0 → packed[0] bit 0 set
    CHECK((packed[0] & 0x01) != 0);
    // Bit 7 → byte 0, bit 7 → packed[0] bit 7 set
    CHECK((packed[0] & 0x80) != 0);
    // Bit 15 → byte 1, bit 7 → packed[1] bit 7 set
    CHECK((packed[1] & 0x80) != 0);
}

// ─── metadata functions ──────────────────────────────────────────────────

TEST_CASE("build_metadata_file and build_metadata_map_from_tsv round-trip")
{
    std::string tsv_path = std::filesystem::temp_directory_path() / "test_meta.tsv";

    std::vector<genivf::WindowMetaData> windows;
    windows.push_back({ "seq1", 0 });
    windows.push_back({ "seq1", 10 });
    windows.push_back({ "seq2", 5 });

    REQUIRE(genivf::build_metadata_file(tsv_path, windows));

    std::unordered_map<size_t, genivf::WindowMetaData> map;
    REQUIRE(genivf::build_metadata_map_from_tsv(tsv_path, map));

    CHECK_EQ(map.size(), 3u);
    CHECK_EQ(map[0].sequence_name, "seq1");
    CHECK_EQ(map[0].start_pos, 0);
    CHECK_EQ(map[1].sequence_name, "seq1");
    CHECK_EQ(map[1].start_pos, 10);
    CHECK_EQ(map[2].sequence_name, "seq2");
    CHECK_EQ(map[2].start_pos, 5);

    std::filesystem::remove(tsv_path);
}

TEST_CASE("build_metadata_map_from_tsv: handles empty file")
{
    std::string tsv_path = std::filesystem::temp_directory_path() / "empty.tsv";
    {
        std::ofstream ofs(tsv_path);
        ofs << "index\tsequence_name\tstart_position\n";
    }

    std::unordered_map<size_t, genivf::WindowMetaData> map;
    CHECK(genivf::build_metadata_map_from_tsv(tsv_path, map));
    CHECK(map.empty());

    std::filesystem::remove(tsv_path);
}

TEST_CASE("build_metadata_file: returns false for empty input")
{
    std::vector<genivf::WindowMetaData> empty;
    CHECK_FALSE(genivf::build_metadata_file("/dev/null/not_created", empty));
}

// ─── quantisation round-trip ──────────────────────────────────────────────

TEST_CASE("real_to_binary and binary_to_real round-trip")
{
    constexpr size_t bits = 16; // 2 bytes
    float float_in[16] = {};
    for (size_t i = 0; i < 16; ++i) {
        float_in[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    }

    uint8_t packed[2] = {};
    REQUIRE(genivf::real_to_binary(bits, float_in, packed));

    float float_out[16] = {};
    REQUIRE(genivf::binary_to_real(bits, packed, float_out));

    for (size_t i = 0; i < 16; ++i) {
        CHECK_EQ(float_out[i], (i % 3 == 0) ? 1.0f : -1.0f);
    }
}
