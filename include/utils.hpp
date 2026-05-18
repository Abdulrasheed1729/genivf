#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace genivf {

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

// Computes the Hamming distance between two packed-binary vectors.
[[nodiscard]] inline uint32_t
distance_hamming(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t d = 0;
    for (std::size_t i = 0; i < N; ++i) {
        d += std::popcount(static_cast<uint8_t>(a[i] ^ b[i]));
    }
    return d;
}

// Computes the Jaccard distance between two packed-binary vectors.
[[nodiscard]] inline float
distance_jaccard(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t bits_union        = 0;
    uint32_t bits_intersection = 0;
    for (std::size_t i = 0; i < N; ++i) {
        bits_union +=
          std::popcount(static_cast<uint8_t>(a[i] | b[i]));
        bits_intersection +=
          std::popcount(static_cast<uint8_t>(a[i] & b[i]));
    }
    return bits_union == 0
             ? 0.0f
             : 1.0f - static_cast<float>(bits_intersection) /
                        static_cast<float>(bits_union);
}

// Convert a d-dimensional float vector to a packed binary vector.
[[nodiscard]] inline bool
real_to_binary(std::size_t d, const float* x_in, uint8_t* x_out)
{
    if (d % 8 != 0)
        return false;
    for (std::size_t i = 0; i < d / 8; ++i) {
        uint8_t byte = 0;
        for (int j = 0; j < 8; ++j) {
            if (x_in[i * 8 + j] > 0.0f) {
                byte |= static_cast<uint8_t>(1u << j);
            }
        }
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
    for (std::size_t i = 0; i < d; ++i) {
        x_out[i] =
          2.0f * static_cast<float>((x_in[i >> 3] >> (i & 7)) & 1u) - 1.0f;
    }
    return true;
}

} // namespace genivf
