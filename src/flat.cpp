#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

#include "flat.hpp"
#include "logger.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace genivf {

IndexFlat::IndexFlat(size_t dim, size_t ntotal)
  : d_dim(dim)
  , d_ntotal(ntotal)
{
    d_vectors.reserve(ntotal);
    log::info(
      "Initialized IndexFlat with dim = {} bytes, ntotal = {}", dim, ntotal);
}

void
IndexFlat::add(std::span<const Point> points)
{
    log::info("Adding {} points to the index...", points.size());

    for (const auto& point : points) {
        if (point.values.size() != d_dim) {
            throw std::invalid_argument(
              "IndexFlat::add: point dimension does not match index dimension");
        }

        auto [it, inserted] = d_vectors.emplace(point.id, point);
        if (!inserted) {
            throw std::invalid_argument("IndexFlat::add: duplicate point id " +
                                        std::to_string(point.id));
        }

        log::debug("Added point ID {}", point.id);
    }
    d_ntotal = d_vectors.size();
    log::info("Successfully added {} points to the index.", points.size());
}

template<MetricType Metric>
std::vector<SearchResult>
IndexFlat::search_impl(const Point& query, size_t k) const
{
    std::vector<SearchResult> candidates;
    candidates.reserve(d_ntotal);

    for (const auto& [id, point] : d_vectors) {
        double dist = 0.0;
        if constexpr (Metric == MetricType::L2) {
            dist = distance_l2(query.values.data(), point.values.data(), d_dim);
        } else if constexpr (Metric == MetricType::HAMMING) {
            dist = static_cast<double>(distance_hamming(
              query.values.data(), point.values.data(), d_dim));
        } else if constexpr (Metric == MetricType::JACCARD) {
            dist = static_cast<double>(distance_jaccard(
              query.values.data(), point.values.data(), d_dim));
        }
        candidates.push_back({ id, dist });
    }

    const size_t total_scanned = candidates.size();
    const size_t result_count = std::min(k, total_scanned);
    std::partial_sort(candidates.begin(),
                      candidates.begin() +
                        static_cast<std::ptrdiff_t>(result_count),
                      candidates.end());
    candidates.resize(result_count);

    log::info("Search complete. Scanned {} candidates, returning top {}",
              total_scanned,
              result_count);

    return candidates;
}

std::vector<SearchResult>
IndexFlat::search(const Point& query,
                  size_t k,
                  size_t nprobe,
                  MetricType metric) const
{
    log::info("Executing Search query: k = {}, nprobe = {}, metric = {}",
              k,
              nprobe,
              static_cast<int>(metric));

    if (query.values.size() != d_dim) {
        throw std::invalid_argument(
          "IndexFlat::search: query dimension does not match index dimension");
    }
    if (k == 0) {
        return {};
    }

    switch (metric) {
        case MetricType::L2:
            return search_impl<MetricType::L2>(query, k);
        case MetricType::HAMMING:
            return search_impl<MetricType::HAMMING>(query, k);
        case MetricType::JACCARD:
            return search_impl<MetricType::JACCARD>(query, k);
    }
    return {};
}

} // namespace genivf
