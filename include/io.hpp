#pragma once

#include <bit>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

#include "flat.hpp"
#include "genivf.hpp"

namespace genivf::io {

inline constexpr uint8_t kMagic[4] = { 0x47, 0x49, 0x56, 0x46 };
inline constexpr uint8_t kVersion = 1;

namespace detail {

// NOTE: Reverses byte order of T if the host is big-endian so that all values
// are written to disk in little-endian order. On little-endian hosts
// (x86/ARM-LE) the compiler folds this to a no-op.
template<typename T>
[[nodiscard]] T
to_le(T value) noexcept
{
    if constexpr (std::endian::native == std::endian::little) {
        return value;
    } else {
        return std::byteswap(value); // C++23, <bit>
    }
}

// Symmetric: LE-on-disk → native.
template<typename T>
[[nodiscard]] T
from_le(T value) noexcept
{
    return to_le(value);
}

inline void
write_bytes(std::ostream& os, const void* data, std::size_t n)
{
    os.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!os) {
        throw std::runtime_error("genivf::io: write error");
    }
}

template<typename T>
void
write_val(std::ostream& os, T value)
{
    const T le = to_le(value);
    write_bytes(os, &le, sizeof(T));
}

inline void
read_bytes(std::istream& is, void* data, std::size_t n)
{
    is.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
    if (!is || static_cast<std::size_t>(is.gcount()) != n) {
        throw std::runtime_error(
          "genivf::io: read error or unexpected end of file");
    }
}

template<typename T>
[[nodiscard]] T
read_val(std::istream& is)
{
    T le{};
    read_bytes(is, &le, sizeof(T));
    return from_le(le);
}

} // namespace detail

void
save_index(const IndexIVF&, const std::filesystem::path&);

[[nodiscard]] IndexIVF
load_index(const std::filesystem::path&);

void
save_flat_index(const IndexFlat&, const std::filesystem::path&);

[[nodiscard]] IndexFlat
load_flat_index(const std::filesystem::path&);

} // namespace genivf::io
