#pragma once

#include <cstddef>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace genivf {

struct IndexIVF;
struct IndexFlat;

namespace io {
void
save_index(const IndexIVF&, const std::filesystem::path&);
IndexIVF
load_index(const std::filesystem::path&);
void
save_flat_index(const IndexFlat&, const std::filesystem::path&);
IndexFlat
load_flat_index(const std::filesystem::path&);
} // namespace io

struct IndexFlat
{

    IndexFlat(size_t dim, size_t ntotal);

    void add(std::span<const Point> points);

    [[nodiscard]] std::vector<SearchResult> search(
      const Point& query,
      size_t k,
      size_t nprobe = 1,
      MetricType metric = MetricType::HAMMING) const;

    // IO functions need direct access to private members to serialise and
    // reconstruct the index without exposing them through the public API.
    friend void io::save_flat_index(const IndexFlat&, const std::filesystem::path&);
    friend IndexFlat io::load_flat_index(const std::filesystem::path&);

  private:
    size_t d_dim;
    size_t d_ntotal;
    std::unordered_map<std::size_t, Point> d_vectors;

    template<MetricType Metric>
    [[nodiscard]] std::vector<SearchResult> search_impl(const Point& query,
                                                        size_t k,
                                                        size_t nprobe) const;
};

} // namespace genivf
