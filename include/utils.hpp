#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "seq.hpp"
#include "types.hpp"

namespace genivf {

template<size_t KMER>
static constexpr size_t
kmer_vector_size()
{
    return (size_t(1) << (2u * KMER));
}

inline std::int8_t
base_to_bits(char base) noexcept
{
    switch (base) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return -1;
    }
}

template<size_t KMER_K, size_t KMER_DIM>
inline void
compute_binary_kmer_vector(const char* sequence,
                           size_t len,
                           std::array<std::uint8_t, KMER_DIM>& out)
{
    out.fill(0);
    if (len < KMER_K) {
        return;
    }

    constexpr std::uint32_t MASK =
      static_cast<std::uint32_t>(KMER_DIM - 1); // keep last 2*K bits
    std::uint32_t val = 0;
    int run = 0; // number of consecutive valid bases seen

    for (std::size_t i = 0; i < len; ++i) {
        int code = base_to_bits(sequence[i]);
        if (code < 0) {
            // Break k-mer when encountering non-ACGT.
            run = 0;
            val = 0;
            continue;
        }

        val = ((val << 2) | static_cast<std::uint32_t>(code)) & MASK;
        if (static_cast<size_t>(run) < KMER_K) {
            ++run;
            if (static_cast<size_t>(run) < KMER_K) {
                continue; // haven't formed a full k-mer yet
            }
        }
        out[val] = 1; // binary: mark presence, no counting
    }
}

template<size_t KMER_DIM>
inline void
pack_kmer_vector_endian(const std::array<std::uint8_t, KMER_DIM>& kmer_vector,
                        std::uint8_t* out)
{
    static_assert(KMER_DIM > 0 && KMER_DIM % 8 == 0,
                  "KMER_DIM must be > 0  and divisible by 8");

    const size_t num_bytes = KMER_DIM >> 3;
    for (std::size_t i = 0; i < num_bytes; ++i) {
        out[i] = 0;
    }
    for (std::size_t bit = 0; bit < KMER_DIM; ++bit) {
        if (!kmer_vector[bit])
            continue;
        std::size_t byte_idx = bit / 8;
        std::size_t bit_in_byte = bit % 8;
        out[byte_idx] |= static_cast<std::uint8_t>(1u << (7 - bit_in_byte));
    }
}

template<size_t KMER_K, size_t KMER_DIM>
void
build_kmer_vectors_from_fasta_file(
  const std::string& fasta_file_path,
  std::vector<std::array<uint8_t, KMER_DIM>>& output_vector,
  std::vector<WindowMetaData>& windows,
  size_t window_size = 50,
  size_t stride = 1)
{
    seq::FastaScanner scanner(fasta_file_path);

    while (scanner.hasNext()) {
        seq::FastaRecord record = scanner.next();
        std::string sequence = record.sequence;

        for (size_t i = 0; i + window_size <= sequence.size(); i += stride) {
            std::array<uint8_t, KMER_DIM> v;
            v.fill(0);
            compute_binary_kmer_vector<KMER_K, KMER_DIM>(
              sequence.data() + i, window_size, v);

            output_vector.push_back(std::move(v));
            WindowMetaData meta;
            meta.sequence_name = record.header;
            meta.start_pos = i;
            windows.push_back(std::move(meta));
        }
    }
}

template<size_t KMER_DIM>
bool
build_packed_kmer_vectors(
  const std::vector<std::array<uint8_t, KMER_DIM>>& input_vectors,
  std::vector<uint8_t>& packed_vectors)
{
    if (input_vectors.empty())
        return false;

    size_t nb = input_vectors.size();
    size_t num_of_bytes = KMER_DIM >> 3;

    packed_vectors.resize(nb * num_of_bytes);

    for (size_t i = 0; i < nb; ++i) {
        pack_kmer_vector_endian(input_vectors[i],
                                packed_vectors.data() + i * num_of_bytes);
    }

    if (packed_vectors.empty())
        return false;

    return true;
}

inline bool
build_metadata_file(const std::string& filename,
                    const std::vector<WindowMetaData>& windows)
{
    if (windows.empty())
        return false;
    std::ofstream metafile(filename);
    metafile << "index" << "\t" << "sequence_name" << "\t" << "start_position"
             << "\n";
    for (size_t i = 0; i < windows.size(); ++i) {
        metafile << i << "\t" << windows[i].sequence_name << "\t"
                 << windows[i].start_pos << "\n";
    }

    return true;
}

inline bool
build_metadata_map_from_tsv(const std::string& tsv_file,
                            std::unordered_map<size_t, WindowMetaData>& map)
{
    std::ifstream file(tsv_file);
    if (!file.is_open())
        return false;

    std::string header;

    // skip header
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) {
        size_t idx;
        std::string idx_str;
        std::string start_pos_str;
        std::string sequence_name;
        WindowMetaData meta_data;
        std::istringstream iss(line);
        std::getline(iss, idx_str, '\t');
        std::getline(iss, sequence_name, '\t');
        std::getline(iss, start_pos_str, '\t');
        idx = std::stoul(idx_str.c_str());
        meta_data.start_pos = std::stoi(start_pos_str);
        meta_data.sequence_name = std::move(sequence_name);

        map[idx] = std::move(meta_data);
    }

    return true;
}

template<size_t KMER_K, size_t KMER_DIM>
inline void
build_query_kmer_vectors(const std::string& fastq_file,
                         std::vector<std::vector<std::uint8_t>>& out)
{
    seq::FastqScanner scanner(fastq_file);
    std::array<std::uint8_t, KMER_DIM> arr;
    size_t num_of_bytes = KMER_DIM >> 3;
    std::vector<std::uint8_t> packed_vector_buffer(num_of_bytes, 0);

    while (scanner.hasNext()) {
        seq::FastqRecord record = scanner.next();

        compute_binary_kmer_vector<KMER_K, KMER_DIM>(
          record.sequence.c_str(), record.sequence.length(), arr);
        pack_kmer_vector_endian(arr, packed_vector_buffer.data());
        out.push_back(packed_vector_buffer);
    }
}

// NOTE: =====Distance Functions ========

// Returns the squared Euclidean distance between two byte arrays of length N.
[[nodiscard]] inline double
distance_l2_sq(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double diff =
          static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

[[nodiscard]] inline double
distance_l2(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    return std::sqrt(distance_l2_sq(a, b, N));
}

// Computes the Hamming distance between two packed-binary vectors using 64-bit
// words.
[[nodiscard]] inline uint32_t
distance_hamming(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t d = 0;
    std::size_t i = 0;

    // Process in 64-bit (8-byte) chunks to leverage fast CPU instructions
    // INFO: will continue to  use `std::popcount` for now since not all compilers support
    // `__builtin_popcount`
    const std::size_t num_words = N / 8;
    if (num_words > 0) {
        // TODO: I think as part of future work, the bit packing should be in 64-bit by default?
        // Hmm... there is a natural simd support in cpp 26, hopefully in the future will change to that
        const uint64_t* a_64 = reinterpret_cast<const uint64_t*>(a);
        const uint64_t* b_64 = reinterpret_cast<const uint64_t*>(b);
        for (std::size_t w = 0; w < num_words; ++w) {
            d += std::popcount(a_64[w] ^ b_64[w]);
        }
        i = num_words * 8;
    }

    // Process remaining bytes
    for (; i < N; ++i) {
        d += std::popcount(static_cast<uint8_t>(a[i] ^ b[i]));
    }
    return d;
}

// Computes the Jaccard distance between two packed-binary vectors using 64-bit
// words.
[[nodiscard]] inline float
distance_jaccard(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t bits_union = 0;
    uint32_t bits_intersection = 0;
    std::size_t i = 0;

    // Process in 64-bit (8-byte) chunks
    const std::size_t num_words = N / 8;
    if (num_words > 0) {
        const uint64_t* a_64 = reinterpret_cast<const uint64_t*>(a);
        const uint64_t* b_64 = reinterpret_cast<const uint64_t*>(b);
        for (std::size_t w = 0; w < num_words; ++w) {
            bits_union += std::popcount(a_64[w] | b_64[w]);
            bits_intersection += std::popcount(a_64[w] & b_64[w]);
        }
        i = num_words * 8;
    }

    // Process remaining bytes
    for (; i < N; ++i) {
        bits_union += std::popcount(static_cast<uint8_t>(a[i] | b[i]));
        bits_intersection += std::popcount(static_cast<uint8_t>(a[i] & b[i]));
    }

    return bits_union == 0 ? 0.0f
                           : 1.0f - static_cast<float>(bits_intersection) /
                                      static_cast<float>(bits_union);
}

// NOTE: === Quantisation Functions ===

// Convert a d-dimensional float vector to a packed binary vector of dimension
// d/8.
[[nodiscard]] inline bool
real_to_binary(std::size_t d, const float* x_in, uint8_t* x_out)
{
    if (d % 8 != 0)
        return false;

    const std::size_t n_bytes = d / 8;
    for (std::size_t i = 0; i < n_bytes; ++i) {
        uint8_t byte = 0;
        for (std::size_t j = 0; j < 8; ++j)
            byte |=
              static_cast<uint8_t>((x_in[i * 8 + j] > 0.0f ? 1u : 0u) << j);
        x_out[i] = byte;
    }
    return true;
}

// Expand a packed binary vector into floats (+1.0 / -1.0).
[[nodiscard]] inline bool
binary_to_real(std::size_t d, const uint8_t* x_in, float* x_out)
{
    if (d % 8 != 0)
        return false;

    for (std::size_t i = 0; i < d; ++i)
        x_out[i] = ((x_in[i >> 3] >> (i & 7)) & 1u) ? 1.0f : -1.0f;

    return true;
}

} // namespace genivf
