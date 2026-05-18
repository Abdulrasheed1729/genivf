#include "genivf.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace genivf {

uint8_t
Point::operator[](size_t index) const
{
    if (index >= values.size()) {
        throw std::out_of_range("Point::operator[]: index " +
                                std::to_string(index) + " out of range [0, " +
                                std::to_string(values.size()) + ")");
    }
    return values[index];
}

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

IndexIVF::IndexIVF(size_t num_cells, size_t dim, unsigned seed)
  : d_num_cells(num_cells)
  , d_dim(dim)
  , d_seed(seed)
{
    validate_ivf_params(num_cells, dim);
}

// Delegate to the three-argument constructor to avoid duplicating validation.
IndexIVF::IndexIVF(size_t num_cells, size_t dim)
  : IndexIVF(num_cells, dim, 42u)
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
IndexIVF::train(std::span<const Point> points,
                size_t max_iter,
                double /* epsilon — unused for binary centroids */)
{
    if (points.size() < d_num_cells) {
        throw std::invalid_argument(
          "IndexIVF::train: need at least num_cells training points");
    }
    for (const auto& p : points) {
        if (p.values.size() != d_dim) {
            throw std::invalid_argument("IndexIVF::train: point dimension does "
                                        "not match index dimension");
        }
    }

    // Clear both stores so re-training is safe.
    d_clusters.clear();
    d_vectors.clear();

    const size_t n = points.size();

    // Initialise centroids by randomly sampling k distinct points (uniform
    // random initialisation). K-means++ would give faster convergence and is a
    // natural future improvement.
    std::mt19937 rng(d_seed);
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<Point> centroids;
    centroids.reserve(d_num_cells);
    for (size_t i = 0; i < d_num_cells; ++i) {
        centroids.emplace_back(i, points[idx[i]].values);
    }

    // Each entry holds the indices (into `points`) of the points assigned to
    // cluster i in the current iteration.
    std::vector<std::vector<size_t>> assignments(d_num_cells);

    for (size_t iter = 0; iter < max_iter; ++iter) {

        for (auto& a : assignments) {
            a.clear();
        }

        // Assign each point to its nearest centroid by Hamming distance.
        for (size_t i = 0; i < n; ++i) {
            size_t nearest = 0;
            uint32_t min_dist = distance_hamming(
              points[i].values.data(), centroids[0].values.data(), d_dim);

            for (size_t j = 1; j < d_num_cells; ++j) {
                const uint32_t dist = distance_hamming(
                  points[i].values.data(), centroids[j].values.data(), d_dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = j;
                }
            }
            assignments[nearest].push_back(i);
        }

        // Recompute each centroid as the per-bit majority vote of its assigned
        // points. This is the Fréchet mean under Hamming distance: bit j of
        // the new centroid is 1 iff strictly more than half the assigned
        // points have bit j set.
        //
        // Convergence check: binary centroids are discrete, so movement is
        // measured in Hamming distance. Any bit change means not converged.
        bool converged = true;

        for (size_t i = 0; i < d_num_cells; ++i) {
            if (assignments[i].empty()) {
                // Cluster lost all members — keep the old centroid in place.
                // Alternative: re-seed from the highest-error point
                // (as done in some FAISS variants).
                continue;
            }

            const size_t count = assignments[i].size();
            // Accumulate per-bit set-counts across all assigned points.
            // d_dim bytes × 8 bits/byte = total binary dimensions.
            std::vector<uint32_t> bit_counts(d_dim * 8, 0u);

            for (const size_t pt_idx : assignments[i]) {
                for (size_t byte_pos = 0; byte_pos < d_dim; ++byte_pos) {
                    const uint8_t byte_val = points[pt_idx].values[byte_pos];
                    for (int bit = 0; bit < 8; ++bit) {
                        bit_counts[byte_pos * 8 + bit] +=
                          static_cast<uint32_t>((byte_val >> bit) & 1u);
                    }
                }
            }

            // Pack the majority-vote result back into bytes.
            std::vector<uint8_t> new_bytes(d_dim, 0u);
            for (size_t byte_pos = 0; byte_pos < d_dim; ++byte_pos) {
                uint8_t new_byte = 0;
                for (int bit = 0; bit < 8; ++bit) {
                    if (bit_counts[byte_pos * 8 + bit] * 2 > count) {
                        new_byte |= static_cast<uint8_t>(1u << bit);
                    }
                }
                new_bytes[byte_pos] = new_byte;
            }

            if (distance_hamming(
                  centroids[i].values.data(), new_bytes.data(), d_dim) != 0u) {
                converged = false;
            }

            centroids[i].values = std::move(new_bytes);
        }

        if (converged) {
            break;
        }
    }

    // Materialise clusters with empty inverted lists; add() will fill them.
    d_clusters.reserve(d_num_cells);
    for (size_t i = 0; i < d_num_cells; ++i) {
        d_clusters.emplace_back(i, std::move(centroids[i]));
    }
}

void
IndexIVF::add(std::span<const Point> points)
{
    if (!is_trained()) {
        throw std::logic_error("IndexIVF::add: call train() before add()");
    }

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
    }
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

    // partial_sort is O(C log nprobe) — cheaper than a full sort when
    // nprobe << num_cells.
    std::vector<std::pair<double, size_t>> centroid_dists;
    centroid_dists.reserve(d_clusters.size());

    for (size_t i = 0; i < d_clusters.size(); ++i) {
        double dist = 0.0;
        switch (metric) {
            case MetricType::L2:
                dist = distance_l2_sq(query.values.data(),
                                      d_clusters[i].centroid.values.data(),
                                      d_dim);
                break;
            case MetricType::HAMMING:
                dist = static_cast<double>(
                  distance_hamming(query.values.data(),
                                   d_clusters[i].centroid.values.data(),
                                   d_dim));
                break;
            case MetricType::JACCARD:
                dist = static_cast<double>(
                  distance_jaccard(query.values.data(),
                                   d_clusters[i].centroid.values.data(),
                                   d_dim));
                break;
        }
        centroid_dists.emplace_back(dist, i);
    }

    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() +
                        static_cast<std::ptrdiff_t>(nprobe),
                      centroid_dists.end());

    std::vector<SearchResult> candidates;

    for (size_t p = 0; p < nprobe; ++p) {
        const size_t cell_idx = centroid_dists[p].second;

        for (const size_t point_id : d_clusters[cell_idx].point_indices) {
            const Point& pt = d_vectors.at(point_id);
            double dist = 0.0;
            switch (metric) {
                case MetricType::L2:
                    dist =
                      distance_l2(query.values.data(), pt.values.data(), d_dim);
                    break;
                case MetricType::HAMMING:
                    dist = static_cast<double>(distance_hamming(
                      query.values.data(), pt.values.data(), d_dim));
                    break;
                case MetricType::JACCARD:
                    dist = static_cast<double>(distance_jaccard(
                      query.values.data(), pt.values.data(), d_dim));
                    break;
            }
            candidates.push_back({ point_id, dist });
        }
    }

    const size_t result_count = std::min(k, candidates.size());
    std::partial_sort(candidates.begin(),
                      candidates.begin() +
                        static_cast<std::ptrdiff_t>(result_count),
                      candidates.end());
    candidates.resize(result_count);
    return candidates;
}

} // namespace genivf
