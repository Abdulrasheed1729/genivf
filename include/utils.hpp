#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <print>

namespace genivf {

namespace log {

enum class Level {
    NONE = 0,
    INFO = 1,
    DEBUG = 2
};

inline Level g_level = Level::INFO;

inline void set_level(Level level) noexcept {
    g_level = level;
}

template <typename... Args>
inline void info(std::format_string<Args...> fmt, Args&&... args) {
    if (g_level >= Level::INFO) {
        std::print(std::clog, "[GENIVF INFO] ");
        std::println(std::clog, fmt, std::forward<Args>(args)...);
    }
}

template <typename... Args>
inline void debug(std::format_string<Args...> fmt, Args&&... args) {
    if (g_level >= Level::DEBUG) {
        std::print(std::clog, "[GENIVF DEBUG] ");
        std::println(std::clog, fmt, std::forward<Args>(args)...);
    }
}

} // namespace log

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

// Computes the Hamming distance between two packed-binary vectors using 64-bit words.
[[nodiscard]] inline uint32_t
distance_hamming(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t d = 0;
    std::size_t i = 0;

    // Process in 64-bit (8-byte) chunks to leverage fast CPU instructions
    const std::size_t num_words = N / 8;
    if (num_words > 0) {
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

// Computes the Jaccard distance between two packed-binary vectors using 64-bit words.
[[nodiscard]] inline float
distance_jaccard(const uint8_t* a, const uint8_t* b, std::size_t N)
{
    uint32_t bits_union        = 0;
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
