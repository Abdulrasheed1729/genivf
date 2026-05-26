#include "ivf.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cassert>
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

IndexIVF::IndexIVF(size_t num_cells,
                   size_t dim,
                   unsigned seed,
                   InitType init)
  : d_num_cells(num_cells)
  , d_dim(dim)
  , d_seed(seed)
  , d_init_type(init)
{
    validate_ivf_params(num_cells, dim);
    log::info("IndexIVF constructed: init = {}",
              static_cast<int>(d_init_type));
}

// Delegate to the four-argument constructor to avoid duplicating validation.
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

void
IndexIVF::train(std::span<const Point> points, size_t max_iter, double epsilon)
{
    if (points.size() < d_num_cells) {
        throw std::invalid_argument(
          "IndexIVF::train: need at least num_cells training points");
    }

    const size_t n = points.size();
    const size_t num_bits = d_dim * 8;

    log::info("Training IndexIVF with {} cells, dim = {} bytes ({} bits) on {} "
              "points (max_iter = {}, epsilon = {:.6f})",
              d_num_cells,
              d_dim,
              num_bits,
              n,
              max_iter,
              epsilon);

    // 1. Unpack binary training points into float vectors
    log::info(
      "Unpacking binary vectors into continuous floating-point space...");
    std::vector<float> float_points(n * num_bits);
    for (size_t i = 0; i < n; ++i) {
        if (points[i].values.size() != d_dim) {
            throw std::invalid_argument("IndexIVF::train: point dimension does "
                                        "not match index dimension");
        }
        if (!binary_to_real(
              num_bits, points[i].values.data(), &float_points[i * num_bits])) {
            throw std::runtime_error(
              "IndexIVF::train: failed to unpack binary vector to floats");
        }
    }

    // Clear both stores so re-training is safe.
    d_clusters.clear();
    d_vectors.clear();

    // Squared L2 calculation lambda (used by both initialisation and k-means)
    auto get_l2_sq = [num_bits](const float* a, const float* b) noexcept {
        float sum = 0.0f;
        for (size_t d = 0; d < num_bits; ++d) {
            const float diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    };

    // 2. Initialise float centroids (RANDOM or K-MEANS++)
    std::vector<float> float_centroids(d_num_cells * num_bits);
    std::mt19937 rng(d_seed);

    if (d_init_type == InitType::RANDOM) {
        log::info("Initializing centroids via random uniform sampling...");
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::ranges::shuffle(idx, rng);
        for (size_t i = 0; i < d_num_cells; ++i) {
            std::copy_n(&float_points[idx[i] * num_bits],
                        num_bits,
                        &float_centroids[i * num_bits]);
        }
    } else {
        log::info("Initializing centroids via k-means++ ...");
        std::vector<double> min_dists_sq(n,
                                         std::numeric_limits<double>::max());
        std::vector<bool> used(n, false);

        std::uniform_int_distribution<size_t> pick(0, n - 1);
        size_t first = pick(rng);
        used[first] = true;
        std::copy_n(&float_points[first * num_bits],
                    num_bits,
                    &float_centroids[0]);

        for (size_t c = 1; c < d_num_cells; ++c) {
            double total = 0.0;

            const float* last_ctr = &float_centroids[(c - 1) * num_bits];
            for (size_t i = 0; i < n; ++i) {
                if (used[i])
                    continue;
                double d = static_cast<double>(
                  get_l2_sq(&float_points[i * num_bits], last_ctr));
                if (d < min_dists_sq[i]) {
                    min_dists_sq[i] = d;
                }
                total += min_dists_sq[i];
            }

            if (total == 0.0) {
                std::vector<size_t> unused;
                for (size_t i = 0; i < n; ++i) {
                    if (!used[i])
                        unused.push_back(i);
                }
                std::ranges::shuffle(unused, rng);
                for (size_t r = c; r < d_num_cells && (r - c) < unused.size();
                     ++r) {
                    std::copy_n(&float_points[unused[r - c] * num_bits],
                                num_bits,
                                &float_centroids[r * num_bits]);
                }
                break;
            }

            std::uniform_real_distribution<double> draw(0.0, total);
            double threshold = draw(rng);
            double cumulative = 0.0;
            size_t next = 0;
            for (size_t i = 0; i < n; ++i) {
                if (used[i])
                    continue;
                cumulative += min_dists_sq[i];
                if (cumulative >= threshold) {
                    next = i;
                    break;
                }
            }

            used[next] = true;
            std::copy_n(&float_points[next * num_bits],
                        num_bits,
                        &float_centroids[c * num_bits]);
        }
    }

    // 3. K-Means Main Loop
    std::vector<std::vector<size_t>> assignments(d_num_cells);
    const double epsilon_sq = epsilon * epsilon;
    log::info("Starting K-means iterations...");
    for (size_t iter = 0; iter < max_iter; ++iter) {
        for (auto& a : assignments) {
            a.clear();
        }

        // Assignment step (Squared L2 Distance argmin)
        for (size_t i = 0; i < n; ++i) {
            const float* pt = &float_points[i * num_bits];
            size_t nearest = 0;
            float min_dist = get_l2_sq(pt, &float_centroids[0]);

            for (size_t j = 1; j < d_num_cells; ++j) {
                const float dist =
                  get_l2_sq(pt, &float_centroids[j * num_bits]);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = j;
                }
            }
            assignments[nearest].push_back(i);
        }

        // Update step & Convergence check
        bool converged = true;
        size_t active_clusters = 0;

        for (size_t i = 0; i < d_num_cells; ++i) {
            if (assignments[i].empty()) {
                log::debug("Iteration {}: Cell {} had 0 assignments; keeping "
                           "old centroid",
                           iter,
                           i);
                continue;
            }
            active_clusters++;

            const size_t count = assignments[i].size();
            std::vector<float> new_centroid(num_bits, 0.0f);

            // Accumulate
            for (const size_t pt_idx : assignments[i]) {
                const float* pt = &float_points[pt_idx * num_bits];
                for (size_t d = 0; d < num_bits; ++d) {
                    new_centroid[d] += pt[d];
                }
            }

            // Average
            for (size_t d = 0; d < num_bits; ++d) {
                new_centroid[d] /= static_cast<float>(count);
            }

            // Calculate shift distance
            float shift = 0.0f;
            const float* old_c = &float_centroids[i * num_bits];
            for (size_t d = 0; d < num_bits; ++d) {
                float diff = old_c[d] - new_centroid[d];
                shift += diff * diff;
            }

            if (shift > epsilon_sq) {
                converged = false;
            }

            // Update
            std::ranges::copy(new_centroid,
                      &float_centroids[i * num_bits]);
        }

        log::info("Iteration {:2d}: Centroids updated ({} active cells). "
                  "Movement converged = {}",
                  iter,
                  active_clusters,
                  converged);

        if (converged) {
            log::info("K-Means training converged early at iteration {}.",
                      iter + 1);
            break;
        }
    }

    // 4. Binarize float centroids and store them in d_clusters
    log::info("Binarizing continuous centroids back to packed binary space...");
    d_clusters.reserve(d_num_cells);
    for (size_t i = 0; i < d_num_cells; ++i) {
        std::vector<uint8_t> packed_centroid(d_dim, 0u);
        if (!real_to_binary(num_bits,
                            &float_centroids[i * num_bits],
                            packed_centroid.data())) {
            throw std::runtime_error(
              "IndexIVF::train: failed to pack float centroid to binary");
        }
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

        // emplace() silently no-ops on a key collision, which would leave the
        // inverted list pointing at the old vector — detect it explicitly.
        auto [it, inserted] = d_vectors.emplace(point.id, point);
        if (!inserted) {
            throw std::invalid_argument("IndexIVF::add: duplicate point id " +
                                        std::to_string(point.id));
        }

        const size_t cell = find_nearest_centroid(point);
        d_clusters[cell].point_indices.push_back(point.id);

        // Contiguous layout: append values directly to flat_vectors of the
        // cluster
        d_clusters[cell].flat_vectors.insert(
          d_clusters[cell].flat_vectors.end(),
          point.values.begin(),
          point.values.end());

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

    std::ranges::partial_sort(centroid_dists,centroid_dists.begin() +
                        static_cast<std::ptrdiff_t>(nprobe));

    std::vector<SearchResult> candidates;

    for (size_t p = 0; p < nprobe; ++p) {
        const size_t cell_idx = centroid_dists[p].second;
        const auto& cluster = d_clusters[cell_idx];

        log::debug("Probing cell rank {}: Cell ID = {} (centroid ID = {}, dist "
                   "= {:.4f}), containing {} vectors",
                   p,
                   cell_idx,
                   cluster.centroid.id,
                   centroid_dists[p].first,
                   cluster.point_indices.size());

        for (size_t i = 0; i < cluster.point_indices.size(); ++i) {
            const size_t point_id = cluster.point_indices[i];
            const uint8_t* vector_data = &cluster.flat_vectors[i * d_dim];

            double dist = 0.0;
            if constexpr (Metric == MetricType::L2) {
                dist = distance_l2(query.values.data(), vector_data, d_dim);
            } else if constexpr (Metric == MetricType::HAMMING) {
                dist = static_cast<double>(
                  distance_hamming(query.values.data(), vector_data, d_dim));
            } else if constexpr (Metric == MetricType::JACCARD) {
                dist = static_cast<double>(
                  distance_jaccard(query.values.data(), vector_data, d_dim));
            }
            candidates.push_back({ point_id, dist });
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
