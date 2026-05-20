#pragma once

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "genivf.hpp"

namespace genivf::io {

inline constexpr uint8_t kMagic[4] = { 0x47, 0x49, 0x56, 0x46 };
inline constexpr uint8_t kVersion = 1;

namespace detail {

// NOTE: Reverses byte order of T if the host is big-endian so that all values are
// written to disk in little-endian order. On little-endian hosts (x86/ARM-LE)
// the compiler folds this to a no-op.
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

// Serialises `index` to `path` in the GIVF v1 binary format.
//
// Throws `std::invalid_argument` if the index has not been trained.
// Throws `std::runtime_error` on any I/O failure.
inline void
save_index(const IndexIVF& index, const std::filesystem::path& path)
{
    if (!index.is_trained()) {
        throw std::invalid_argument(
          "genivf::io::save_index: index must be trained before saving");
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error(
          "genivf::io::save_index: cannot open file for writing: " +
          path.string());
    }

    // NOTE: magic bytes are written first to allow detection of corrupted files
    // Magic bytes are ASCII "GIVF" (0x47 0x49 0x56 0x46)
    // Version is written next to allow detection of incompatible file formats
    // Number of cells, dimension, and seed are written next to define the index structure
    // Number of vectors is written next to define the index size
    // Clusters are written next to define the index contents
    detail::write_bytes(ofs, kMagic, sizeof(kMagic));
    detail::write_val<uint8_t>(ofs, kVersion);
    detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(index.d_num_cells));
    detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(index.d_dim));
    detail::write_val<uint32_t>(ofs, static_cast<uint32_t>(index.d_seed));
    detail::write_val<uint64_t>(ofs,
                                static_cast<uint64_t>(index.d_vectors.size()));

    for (const Cluster& cluster : index.d_clusters) {
        detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(cluster.id));
        detail::write_val<uint64_t>(ofs,
                                    static_cast<uint64_t>(cluster.centroid.id));
        detail::write_bytes(ofs, cluster.centroid.values.data(), index.d_dim);

        detail::write_val<uint64_t>(
          ofs, static_cast<uint64_t>(cluster.point_indices.size()));
        for (const size_t pid : cluster.point_indices) {
            detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(pid));
        }
    }

    for (const auto& [id, point] : index.d_vectors) {
        detail::write_val<uint64_t>(ofs, static_cast<uint64_t>(id));
        detail::write_bytes(ofs, point.values.data(), index.d_dim);
    }
}

// Deserialises and returns an IndexIVF from a GIVF v1 binary file at `path`.
// The returned index is fully trained and populated — search() can be called
// immediately.
//
// Throws `std::runtime_error` on any I/O failure, bad magic, unsupported
//   version, or internal consistency violation.
[[nodiscard]] inline IndexIVF
load_index(const std::filesystem::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("genivf::io::load_index: cannot open file: " +
                                 path.string());
    }

    uint8_t magic[4]{};
    detail::read_bytes(ifs, magic, sizeof(magic));
    if (magic[0] != kMagic[0] || magic[1] != kMagic[1] ||
        magic[2] != kMagic[2] || magic[3] != kMagic[3]) {
        throw std::runtime_error(
          "genivf::io::load_index: invalid magic — not a GIVF file");
    }

    const uint8_t version = detail::read_val<uint8_t>(ifs);
    if (version != kVersion) {
        throw std::runtime_error(
          "genivf::io::load_index: unsupported file version " +
          std::to_string(version) + " (expected " + std::to_string(kVersion) +
          ")");
    }

    const size_t num_cells =
      static_cast<size_t>(detail::read_val<uint64_t>(ifs));
    const size_t dim = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
    const unsigned seed =
      static_cast<unsigned>(detail::read_val<uint32_t>(ifs));
    const size_t num_vectors =
      static_cast<size_t>(detail::read_val<uint64_t>(ifs));

    // Construct a fresh, untrained index and populate its private fields
    // directly (friend access).
    IndexIVF index(num_cells, dim, seed);

    index.d_clusters.reserve(num_cells);
    for (size_t c = 0; c < num_cells; ++c) {
        const size_t cluster_id =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        const size_t centroid_id =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));

        std::vector<uint8_t> centroid_vals(dim);
        detail::read_bytes(ifs, centroid_vals.data(), dim);

        Cluster cluster(cluster_id,
                        Point(centroid_id, std::move(centroid_vals)));

        const size_t list_len =
          static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        cluster.point_indices.reserve(list_len);
        for (size_t i = 0; i < list_len; ++i) {
            cluster.point_indices.push_back(
              static_cast<size_t>(detail::read_val<uint64_t>(ifs)));
        }

        index.d_clusters.push_back(std::move(cluster));
    }

    // Sanity-check: we must have read exactly num_cells clusters.
    if (index.d_clusters.size() != num_cells) {
        throw std::runtime_error(
          "genivf::io::load_index: cluster count mismatch in file");
    }

    index.d_vectors.reserve(num_vectors);
    for (size_t v = 0; v < num_vectors; ++v) {
        const size_t pid = static_cast<size_t>(detail::read_val<uint64_t>(ifs));
        std::vector<uint8_t> vals(dim);
        detail::read_bytes(ifs, vals.data(), dim);
        index.d_vectors.emplace(pid, Point(pid, std::move(vals)));
    }

    if (index.d_vectors.size() != num_vectors) {
        throw std::runtime_error(
          "genivf::io::load_index: vector count mismatch in file");
    }

    // Populate the contiguous flat_vectors buffers in d_clusters from the loaded d_vectors
    for (auto& cluster : index.d_clusters) {
        cluster.flat_vectors.resize(cluster.point_indices.size() * dim);
        for (size_t i = 0; i < cluster.point_indices.size(); ++i) {
            const size_t pid = cluster.point_indices[i];
            auto it = index.d_vectors.find(pid);
            if (it == index.d_vectors.end()) {
                throw std::runtime_error(
                  "genivf::io::load_index: referential integrity violation — cluster references non-existent point ID " +
                  std::to_string(pid));
            }
            std::copy(it->second.values.begin(), it->second.values.end(),
                      &cluster.flat_vectors[i * dim]);
        }
    }

    return index;
}

} // namespace genivf::io
