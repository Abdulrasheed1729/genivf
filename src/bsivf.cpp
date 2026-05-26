#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "bsivf.hpp"
#include "ivf.hpp"
#include "logger.hpp"
#include "types.hpp"

namespace genivf {

IndexBSIVF::IndexBSIVF(size_t dim, size_t ntotal, const size_t stride)
  : d_dim(dim)
  , d_ntotal(ntotal)
  , stride(stride)
{
    d_vectors.reserve(ntotal);
    log::info(
      "Initialized IndexBSIVF with dim = {} bytes, ntotal = {}", dim, ntotal);
}

void
IndexBSIVF::construct_centroids()
{
    log::info("Constructing centroids with stride {} ...", this->stride);
    for (size_t i = 0; i + this->stride < this->d_ntotal; ++i) {
        this->centroids.push_back(i * this->stride);
    }

    log::info("Finished constructing {} centroids", centroids.size());
}

std::pair<size_t, size_t>
IndexBSIVF::find_nearest_centroid(const Point& query) const
{
    log::info("Searching for the closest centroid to query...");
    size_t pos = 0;
    size_t best_distance = std::numeric_limits<size_t>::max();

    for (const auto& i : this->centroids) {
        if (i > this->d_ntotal) break;
        auto dist = distance_hamming(this->d_vectors[i], query);
        if (dist < best_distance) {
            best_distance = dist;
            pos = i;
        }
    }

    log::info(
      "Found the best centroid at {} with distance {}", pos, best_distance);

    return { pos, best_distance };
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
IndexBSIVF::search_impl(const Point& query) const
{
    auto compute_dist = [&](const Point& a, const Point& b) -> double {
        if constexpr (Metric == MetricType::L2) {
            return distance_l2(a, b);
        } else if constexpr (Metric == MetricType::HAMMING) {
            return static_cast<double>(distance_hamming(a, b));
        } else if constexpr (Metric == MetricType::JACCARD) {
            return static_cast<double>(distance_jaccard(a, b));
        }
        else return 0.0;
    };

    auto [pos, _] = this->find_nearest_centroid(query);
    double best_distance = compute_dist(query, this->d_vectors[pos]);

    SearchResult candidate{};
    candidate.distance = best_distance;
    candidate.id = this->d_vectors[pos].id;

    size_t s = this->stride;

    while (s > 1) {
        s /= 2;

        if (pos >= s) {
            double dist = compute_dist(query, this->d_vectors[pos - s]);
            if (dist < best_distance) {
                best_distance = dist;
                candidate.distance = dist;
                candidate.id = this->d_vectors[pos - s].id;
                pos = pos - s;
                continue;
            }
        }

        if (pos + s < this->d_vectors.size()) {
            if (const double dist = compute_dist(query, this->d_vectors[pos + s]);dist < best_distance) {
                best_distance = dist;
                candidate.distance = dist;
                candidate.id = this->d_vectors[pos + s].id;
                pos = pos + s;
            }
        }
    }

    return candidate;
}

SearchResult
IndexBSIVF::search(const Point& query, MetricType metric) const
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
            return search_impl<MetricType::L2>(query);
        case MetricType::HAMMING:
            return search_impl<MetricType::HAMMING>(query);
        case MetricType::JACCARD:
            return search_impl<MetricType::JACCARD>(query);
    }
    return {};
}

} // namespace genivf
