#include "ivf.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace genivf {

static void
validate_ivf_params(size_t num_cells, size_t dim)
{
    if (num_cells == 0) {
        throw std::invalid_argument("IndexIVF: num_cells must be > 0");
    }
    if (dim == 0) {
        throw std::invalid_argument("IndexIVF: dim must be > 0");
    }
}

IndexIVF::IndexIVF(size_t num_cells, size_t dim, unsigned seed, InitType init)
  : d_num_cells(num_cells)
  , d_dim(dim)
  , d_seed(seed)
  , d_init_type(init)
{
    validate_ivf_params(num_cells, dim);
    log::info("IndexIVF constructed: init = {}", static_cast<int>(d_init_type));
}

IndexIVF::IndexIVF(size_t num_cells, size_t dim, InitType init)
  : IndexIVF(num_cells, dim, 42u, init)
{
}

size_t
IndexIVF::find_nearest_centroid(const Point& point) const
{
    assert(!d_clusters.empty());

    size_t nearest = 0;
    uint32_t min_dist = distance_hamming(
      point.values.data(), d_clusters[0].centroid.values.data(), d_dim);

    for (size_t i = 1; i < d_clusters.size(); ++i) {
        const uint32_t dist = distance_hamming(
          point.values.data(), d_clusters[i].centroid.values.data(), d_dim);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }
    return nearest;
}

static void
majority_vote_centroid(const std::vector<Point>& points,
                       const std::vector<size_t>& indices,
                       size_t dim,
                       uint8_t* out_centroid)
{
    const size_t num_bits = dim * 8;
    std::vector<uint32_t> counts(num_bits, 0);

    for (const size_t idx : indices) {
        const uint8_t* data = points[idx].values.data();
        for (size_t b = 0; b < num_bits; ++b) {
            counts[b] += (data[b >> 3] >> (b & 7)) & 1u;
        }
    }

    std::memset(out_centroid, 0, dim);
    const uint32_t half = static_cast<uint32_t>(indices.size()) / 2;
    for (size_t b = 0; b < num_bits; ++b) {
        if (counts[b] > half) {
            out_centroid[b >> 3] |= static_cast<uint8_t>(1u << (b & 7));
        }
    }
}

void
IndexIVF::train(std::span<const Point> points, size_t max_iter, double epsilon)
{
    if (points.size() < d_num_cells) {
        throw std::invalid_argument(
          "IndexIVF::train: need at least num_cells training points");
    }

    const size_t n = points.size();
    const size_t num_bits = d_dim * 8;
    (void)epsilon; // unused in binary KMeans (convergence is exact match)

    log::info("Training IndexIVF with {} cells, dim = {} bytes ({} bits) on {} "
              "points (max_iter = {})",
              d_num_cells,
              d_dim,
              num_bits,
              n,
              max_iter);

    // Clear old state — keep points until add() is called.
    d_clusters.clear();

    // Build a local copy of training points for KMeans.
    std::vector<Point> train_points;
    train_points.reserve(n);
    for (const auto& pt : points) {
        if (pt.values.size() != d_dim) {
            throw std::invalid_argument("IndexIVF::train: point dimension does "
                                        "not match index dimension");
        }
        train_points.push_back(pt);
    }

    // Initialise centroids (RANDOM or K-MEANS++)
    // Store centroid bytes in a flat array (num_cells * d_dim bytes).
    std::vector<uint8_t> centroid_data(d_num_cells * d_dim, 0);
    std::mt19937 rng(d_seed);

    if (d_init_type == InitType::RANDOM) {
        log::info("Initializing centroids via random uniform sampling...");
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::ranges::shuffle(idx, rng);
        for (size_t i = 0; i < d_num_cells; ++i) {
            std::memcpy(&centroid_data[i * d_dim],
                        train_points[idx[i]].values.data(),
                        d_dim);
        }
    } else {
        log::info("Initializing centroids via k-means++ ...");
        std::vector<uint32_t> min_dists(n,
                                        std::numeric_limits<uint32_t>::max());
        std::vector<bool> used(n, false);

        std::uniform_int_distribution<size_t> pick(0, n - 1);
        size_t first = pick(rng);
        used[first] = true;
        std::memcpy(
          &centroid_data[0], train_points[first].values.data(), d_dim);

        for (size_t c = 1; c < d_num_cells; ++c) {
            uint64_t total = 0;

            const uint8_t* last_ctr = &centroid_data[(c - 1) * d_dim];
            for (size_t i = 0; i < n; ++i) {
                if (used[i])
                    continue;
                const uint32_t d = distance_hamming(
                  train_points[i].values.data(), last_ctr, d_dim);
                if (d < min_dists[i]) {
                    min_dists[i] = d;
                }
                total += min_dists[i];
            }

            if (total == 0) {
                std::vector<size_t> unused;
                for (size_t i = 0; i < n; ++i) {
                    if (!used[i])
                        unused.push_back(i);
                }
                std::ranges::shuffle(unused, rng);
                for (size_t r = c; r < d_num_cells && (r - c) < unused.size();
                     ++r) {
                    std::memcpy(&centroid_data[r * d_dim],
                                train_points[unused[r - c]].values.data(),
                                d_dim);
                }
                break;
            }

            std::uniform_int_distribution<uint64_t> draw(0, total - 1);
            uint64_t threshold = draw(rng);
            uint64_t cumulative = 0;
            size_t next = 0;
            for (size_t i = 0; i < n; ++i) {
                if (used[i])
                    continue;
                cumulative += min_dists[i];
                if (cumulative >= threshold) {
                    next = i;
                    break;
                }
            }

            used[next] = true;
            std::memcpy(&centroid_data[c * d_dim],
                        train_points[next].values.data(),
                        d_dim);
        }
    }

    // Binary K-Means Main Loop (majority-vote centroids, Hamming
    // assignment)
    {
        std::vector<std::vector<size_t>> assignments(d_num_cells);
        std::vector<uint8_t> new_centroid_bytes(d_dim, 0);
        log::info("Starting binary K-means iterations...");
        for (size_t iter = 0; iter < max_iter; ++iter) {
            for (auto& a : assignments) {
                a.clear();
            }

            // Assignment step (Hamming distance)
            for (size_t i = 0; i < n; ++i) {
                size_t nearest = 0;
                uint32_t min_dist = distance_hamming(
                  train_points[i].values.data(), &centroid_data[0], d_dim);
                for (size_t j = 1; j < d_num_cells; ++j) {
                    const uint32_t dist =
                      distance_hamming(train_points[i].values.data(),
                                       &centroid_data[j * d_dim],
                                       d_dim);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = j;
                    }
                }
                assignments[nearest].push_back(i);
            }

            // Update step (majority-vote) & convergence check
            bool converged = true;
            size_t active_clusters = 0;

            for (size_t i = 0; i < d_num_cells; ++i) {
                if (assignments[i].empty()) {
                    log::debug("Iteration {}: Cell {} had 0 assignments; "
                               "reinitializing to random point",
                               iter,
                               i);
                    std::uniform_int_distribution<size_t> pick_idx(0, n - 1);
                    size_t rand_idx = pick_idx(rng);
                    std::memcpy(&centroid_data[i * d_dim],
                                train_points[rand_idx].values.data(),
                                d_dim);
                    converged = false;
                    continue;
                }
                active_clusters++;

                majority_vote_centroid(train_points,
                                       assignments[i],
                                       d_dim,
                                       new_centroid_bytes.data());

                const uint32_t shift = distance_hamming(
                  &centroid_data[i * d_dim], new_centroid_bytes.data(), d_dim);
                if (shift > 0) {
                    converged = false;
                }

                std::memcpy(
                  &centroid_data[i * d_dim], new_centroid_bytes.data(), d_dim);
            }

            log::info("Iteration {:2d}: Centroids updated ({} active cells). "
                      "Converged = {}",
                      iter,
                      active_clusters,
                      converged);

            if (converged) {
                log::info("Binary K-Means training converged at iteration {}.",
                          iter + 1);
                break;
            }
        }
    }

    // Store centroids in d_clusters
    d_clusters.reserve(d_num_cells);
    for (size_t i = 0; i < d_num_cells; ++i) {
        std::vector<uint8_t> packed_centroid(centroid_data.data() + i * d_dim,
                                             centroid_data.data() +
                                               (i + 1) * d_dim);
        d_clusters.emplace_back(i, Point(i, std::move(packed_centroid)));
    }
    log::info("Training completed successfully.");
}

void
IndexIVF::add(std::span<const Point> points)
{
    if (!is_trained()) {
        throw std::logic_error("IndexIVF::add: call train() before add()");
    }

    log::info("Adding {} points to the index...", points.size());

    for (const auto& point : points) {
        if (point.values.size() != d_dim) {
            throw std::invalid_argument(
              "IndexIVF::add: point dimension does not match index dimension");
        }

        auto [it, inserted] = d_point_index.emplace(point.id, d_points.size());
        if (!inserted) {
            throw std::invalid_argument("IndexIVF::add: duplicate point id " +
                                        std::to_string(point.id));
        }

        d_points.push_back(point);

        const size_t cell = find_nearest_centroid(point);
        d_clusters[cell].point_indices.push_back(d_points.size() - 1);

        log::debug("Added point ID {} -> assigned to Cell {}", point.id, cell);
    }
    log::info("Successfully added {} points to the index.", points.size());
}

template<MetricType Metric>
std::vector<SearchResult>
IndexIVF::search_impl(const Point& query, size_t k, size_t nprobe) const
{
    log::debug(
      "search_impl: running compile-time specialization for MetricType = {}",
      static_cast<int>(Metric));

    std::vector<std::pair<double, size_t>> centroid_dists;
    centroid_dists.reserve(d_clusters.size());

    for (size_t i = 0; i < d_clusters.size(); ++i) {
        double dist = 0.0;
        if constexpr (Metric == MetricType::L2) {
            dist = distance_l2_sq(
              query.values.data(), d_clusters[i].centroid.values.data(), d_dim);
        } else if constexpr (Metric == MetricType::HAMMING) {
            dist = static_cast<double>(
              distance_hamming(query.values.data(),
                               d_clusters[i].centroid.values.data(),
                               d_dim));
        } else if constexpr (Metric == MetricType::JACCARD) {
            dist = static_cast<double>(
              distance_jaccard(query.values.data(),
                               d_clusters[i].centroid.values.data(),
                               d_dim));
        }
        centroid_dists.emplace_back(dist, i);
    }

    std::ranges::partial_sort(centroid_dists,
                              centroid_dists.begin() +
                                static_cast<std::ptrdiff_t>(nprobe));

    std::vector<SearchResult> candidates;

    for (size_t p = 0; p < nprobe; ++p) {
        const size_t cell_idx = centroid_dists[p].second;
        const auto& cluster = d_clusters[cell_idx];

        log::debug("Probing cell rank {}: Cell ID = {} (dist = {:.4f}), "
                   "containing {} vectors",
                   p,
                   cell_idx,
                   centroid_dists[p].first,
                   cluster.point_indices.size());

        for (const size_t point_idx : cluster.point_indices) {
            const auto& pt = d_points[point_idx];
            double dist = 0.0;
            if constexpr (Metric == MetricType::L2) {
                dist =
                  distance_l2(query.values.data(), pt.values.data(), d_dim);
            } else if constexpr (Metric == MetricType::HAMMING) {
                dist = static_cast<double>(distance_hamming(
                  query.values.data(), pt.values.data(), d_dim));
            } else if constexpr (Metric == MetricType::JACCARD) {
                dist = static_cast<double>(distance_jaccard(
                  query.values.data(), pt.values.data(), d_dim));
            }
            candidates.push_back({ pt.id, dist });
        }
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
IndexIVF::search(const Point& query,
                 size_t k,
                 size_t nprobe,
                 MetricType metric) const
{
    if (!is_trained()) {
        throw std::logic_error(
          "IndexIVF::search: call train() before search()");
    }
    if (nprobe == 0 || nprobe > d_clusters.size()) {
        throw std::invalid_argument(
          "IndexIVF::search: nprobe must be in [1, num_cells]");
    }
    if (query.values.size() != d_dim) {
        throw std::invalid_argument(
          "IndexIVF::search: query dimension does not match index dimension");
    }
    if (k == 0) {
        return {};
    }

    log::info("Executing Search query: k = {}, nprobe = {}, metric = {}",
              k,
              nprobe,
              static_cast<int>(metric));

    switch (metric) {
        case MetricType::L2:
            return search_impl<MetricType::L2>(query, k, nprobe);
        case MetricType::HAMMING:
            return search_impl<MetricType::HAMMING>(query, k, nprobe);
        case MetricType::JACCARD:
            return search_impl<MetricType::JACCARD>(query, k, nprobe);
    }
    return {};
}

} // namespace genivf
