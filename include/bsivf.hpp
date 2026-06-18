#pragma once

#include <cstddef>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "types.hpp"

namespace genivf {

struct IndexBSIVF;

struct IndexBSIVF
{

    IndexBSIVF(size_t dim, size_t ntotal);

    void add(std::span<const Point> points);

    [[nodiscard]]
    std::vector<std::pair<size_t, size_t>> find_nearest_centroids(
      const Point& query,
      size_t nprobe = 1) const;

    bool is_trained() const;

    void construct_centroids(size_t stride);

    [[nodiscard]] SearchResult search(
      const Point& query,
      size_t stride = 25,
      size_t min_stride = 1,
      size_t nprobe = 1,
      MetricType metric = MetricType::HAMMING) const;

    [[nodiscard]] size_t num_centroids() const;


    private:
    size_t d_dim;
    size_t d_ntotal;
    std::vector<Point> d_vectors;
    std::unordered_set<size_t> d_ids;
    std::vector<size_t> centroids;

    template<MetricType Metric>
    [[nodiscard]] SearchResult search_impl(const Point& query,
                                           size_t stride = 25,
                                           size_t min_stride = 1,
                                           size_t nprobe = 1) const;
};

} // namespace genivf
