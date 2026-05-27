#pragma once

#include <cstddef>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "types.hpp"

namespace genivf {

struct IndexBSIVF;

// namespace io {
// void
// save_flat_index(const IndexFlat&, const std::filesystem::path&);
// IndexFlat
// load_flat_index(const std::filesystem::path&);
// } // namespace io

struct IndexBSIVF
{

    IndexBSIVF(size_t dim, size_t ntotal, size_t stride);

    void add(std::span<const Point> points);

    [[nodiscard]]
    std::pair<size_t, size_t> find_nearest_centroid(const Point& query) const;

    bool is_trained() const;

    void construct_centroids();

    [[nodiscard]] SearchResult search(
      const Point& query,
      MetricType metric = MetricType::HAMMING) const;

    // friend void io::save_flat_index(const IndexBSIVF&,
    //                                 const std::filesystem::path&);
    // friend IndexBSIVF io::load_flat_index(const std::filesystem::path&);

  private:
    size_t d_dim;
    size_t d_ntotal;
    size_t stride;
    std::vector<Point> d_vectors;
    std::unordered_set<size_t> d_ids;
    std::vector<size_t> centroids;

    template<MetricType Metric>
    [[nodiscard]] SearchResult search_impl(const Point& query) const;
};

} // namespace genivf
