#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "bsivf.hpp"
#include "ivf.hpp"
#include "logger.hpp"
#include "types.hpp"

#include <ranges>

namespace genivf {

IndexBSIVF::IndexBSIVF(size_t dim, size_t ntotal)
  : d_dim(dim)
  , d_ntotal(ntotal)
{
    d_vectors.reserve(ntotal);
    log::info(
      "Initialized IndexBSIVF with dim = {} bytes, ntotal = {}", dim, ntotal);
}

void
IndexBSIVF::construct_centroids(size_t stride)
{
    log::info("Constructing centroids with stride {} ...", stride);
    for (size_t i = 0; i < this->d_ntotal; i += stride) {
        this->centroids.push_back(i);
    }

    log::info("Finished constructing {} centroids", centroids.size());
}

std::vector<std::pair<size_t, size_t>>
IndexBSIVF::find_nearest_centroids(const Point& query, size_t nprobe) const
{
    // Collect all centroid distances
    std::vector<std::pair<size_t, size_t>> centroid_dists;
    centroid_dists.reserve(this->centroids.size());

    for (const auto& i : this->centroids) {
        if (i > this->d_ntotal)
            break;
        auto dist = distance_hamming(this->d_vectors[i], query);
        centroid_dists.emplace_back(i, dist);
    }

    nprobe = std::min(nprobe, centroid_dists.size());
    std::partial_sort(
      centroid_dists.begin(),
      centroid_dists.begin() + static_cast<long>(nprobe),
      centroid_dists.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });

    centroid_dists.resize(nprobe);
    return centroid_dists;
}

bool
IndexBSIVF::is_trained() const
{
    return !this->centroids.empty();
}

void
IndexBSIVF::add(std::span<const Point> points)
{
    log::info("Adding {} points to the index...", points.size());

    for (const auto& point : points) {
        if (point.values.size() != d_dim) {
            throw std::invalid_argument("IndexBSIVF::add: point dimension does "
                                        "not match index dimension");
        }

        auto [it, inserted] = d_ids.insert(point.id);
        if (!inserted) {
            throw std::invalid_argument("IndexBSIVF::add: duplicate point id " +
                                        std::to_string(point.id));
        }

        d_vectors.push_back(point);
        log::debug(
          "Added point ID {} at index {}", point.id, d_vectors.size() - 1);
    }
    d_ntotal = d_vectors.size();
    log::info("Successfully added {} points to the index.", points.size());
}

template<MetricType Metric>
SearchResult
IndexBSIVF::search_impl(const Point& query,
                        const size_t stride,
                        const size_t min_stride,
                        const size_t nprobe) const
{
    assert(min_stride > 0 && stride >= min_stride);

    auto compute_dist = [&](const Point& a, const Point& b) -> double {
        if constexpr (Metric == MetricType::HAMMING)
            return static_cast<double>(distance_hamming(a, b));
        if constexpr (Metric == MetricType::JACCARD)
            return static_cast<double>(distance_jaccard(a, b));
        if constexpr (Metric == MetricType::L2)
            return distance_l2(a, b);
        return 0.0;
    };

    // Get top-N centroid starting positions
    auto centroid_candidates = find_nearest_centroids(query, nprobe);

    SearchResult best{};
    best.distance = std::numeric_limits<double>::max();

    // Track visited indices to avoid redundant distance computations
    // where centroid search regions overlap
    std::unordered_set<size_t> visited;

    for (const auto& pos_start : centroid_candidates | std::views::keys) {
        size_t pos = pos_start;
        double best_distance = compute_dist(query, this->d_vectors[pos]);

        if (visited.insert(pos).second && best_distance < best.distance) {
            best.distance = best_distance;
            best.id = this->d_vectors[pos].id;
        }

        size_t s = stride;

        while (s > min_stride) {
            s /= 2;

            size_t best_pos = pos;

            if (pos >= s) {
                size_t idx = pos - s;
                if (visited.insert(idx).second) {
                    double dist = compute_dist(query, this->d_vectors[idx]);
                    if (dist < best_distance) {
                        best_distance = dist;
                        best_pos = idx;
                        if (dist < best.distance) {
                            best.distance = dist;
                            best.id = this->d_vectors[idx].id;
                        }
                    }
                }
            }

            if (pos + s < this->d_vectors.size()) {
                size_t idx = pos + s;
                if (visited.insert(idx).second) {
                    double dist = compute_dist(query, this->d_vectors[idx]);
                    if (dist < best_distance) {
                        best_distance = dist;
                        best_pos = idx;
                        if (dist < best.distance) {
                            best.distance = dist;
                            best.id = this->d_vectors[idx].id;
                        }
                    }
                }
            }

            pos = best_pos;
        }

        // Linear scan around where this centroid's probe landed
        size_t left = (pos >= min_stride) ? pos - min_stride : 0;
        size_t right = std::min(pos + min_stride + 1, this->d_vectors.size());
        for (size_t i = left; i < right; ++i) {
            if (visited.insert(i).second) {
                double dist = compute_dist(query, this->d_vectors[i]);
                if (dist < best.distance) {
                    best.distance = dist;
                    best.id = this->d_vectors[i].id;
                }
            }
        }
    }

    return best;
}

size_t
IndexBSIVF::num_centroids() const
{
    return this->centroids.size();
}

SearchResult
IndexBSIVF::search(const Point& query,
                   size_t stride,
                   size_t min_stride,
                   size_t nprobe,
                   MetricType metric) const
{
    log::info("Executing Search query: stride = {}, metric = {}",
              stride,
              static_cast<int>(metric));

    if (!is_trained()) {
        throw std::logic_error(
          "IndexBSIVF::search: call construct_centroids() before search()");
    }
    if (query.values.size() != d_dim) {
        throw std::invalid_argument(
          "IndexBSIVF::search: query dimension does not match index dimension");
    }

    switch (metric) {
        case MetricType::L2:
            return search_impl<MetricType::L2>(
              query, stride, min_stride, nprobe);
        case MetricType::HAMMING:
            return search_impl<MetricType::HAMMING>(
              query, stride, min_stride, nprobe);
        case MetricType::JACCARD:
            return search_impl<MetricType::JACCARD>(
              query, stride, min_stride, nprobe);
    }
    return {};
}

} // namespace genivf
