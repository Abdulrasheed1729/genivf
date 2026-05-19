#include "genivf.hpp"

#include <algorithm>
#include <array>
#include <cassert>
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

// Precomputed per-bit lookup: kBitTable[v][b] == (v >> b) & 1u
static constexpr auto kBitTable = []() {
    std::array<std::array<uint8_t, 8>, 256> table{};
    for (int v = 0; v < 256; ++v) {
        for (int b = 0; b < 8; ++b) {
            table[v][b] = static_cast<uint8_t>((v >> b) & 1u);
        }
    }
    return table;
}();

static size_t
nearest_centroid_idx(const Point& point,
                     const std::vector<Point>& centroids,
                     size_t dim)
{
    size_t nearest = 0;
    uint32_t min_dist = distance_hamming(
      point.values.data(), centroids[0].values.data(), dim);

    for (size_t j = 1; j < centroids.size(); ++j) {
        const uint32_t dist = distance_hamming(
          point.values.data(), centroids[j].values.data(), dim);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = j;
        }
    }
    return nearest;
}

void
IndexIVF::train(std::span<const Point> points, size_t max_iter)
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

    d_clusters.clear();
    d_vectors.clear();

    const size_t n = points.size();
    std::mt19937 rng(d_seed);

    std::vector<Point> centroids;
    centroids.reserve(d_num_cells);

    std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
    centroids.push_back(points[idx_dist(rng)]);

    // min_dists[i] = squared Hamming distance to the nearest chosen centroid
    std::vector<uint32_t> min_dists(n, UINT32_MAX);
    for (size_t c = 1; c < d_num_cells; ++c) {
        uint64_t total_weight = 0;

        for (size_t i = 0; i < n; ++i) {
            const uint32_t d = distance_hamming(
              points[i].values.data(), centroids[c - 1].values.data(), d_dim);
            if (d < min_dists[i]) {
                min_dists[i] = d;
            }
            total_weight += static_cast<uint64_t>(min_dists[i]) * min_dists[i];
        }

        if (total_weight == 0) {
            // All remaining points are duplicates of existing centroids.
            centroids.push_back(points[idx_dist(rng)]);
            continue;
        }

        std::uniform_int_distribution<uint64_t> pick(1, total_weight);
        const uint64_t threshold = pick(rng);
        uint64_t cumulative = 0;
        size_t picked = n - 1;
        for (size_t i = 0; i < n; ++i) {
            cumulative += static_cast<uint64_t>(min_dists[i]) * min_dists[i];
            if (cumulative >= threshold) {
                picked = i;
                break;
            }
        }
        centroids.push_back(points[picked]);
    }

    std::vector<std::vector<size_t>> assignments(d_num_cells);
    std::vector<uint32_t> bit_counts(d_dim * 8); // scratch buffer, reused

    for (size_t iter = 0; iter < max_iter; ++iter) {
        for (auto& a : assignments) {
            a.clear();
        }

        // Assign each point to its nearest centroid.
        for (size_t i = 0; i < n; ++i) {
            assignments[nearest_centroid_idx(points[i], centroids, d_dim)]
              .push_back(i);
        }

        bool converged = true;

        for (size_t i = 0; i < d_num_cells; ++i) {
            if (assignments[i].empty()) {
                // Re-seed empty cluster with a random point.
                centroids[i] = points[idx_dist(rng)];
                converged = false;
                continue;
            }

            const size_t count = assignments[i].size();
            std::fill(bit_counts.begin(), bit_counts.end(), 0u);

            // Accumulate per-bit set-counts via precomputed lookup table.
            for (const size_t pt_idx : assignments[i]) {
                for (size_t byte_pos = 0; byte_pos < d_dim; ++byte_pos) {
                    const uint8_t byte_val = points[pt_idx].values[byte_pos];
                    const auto& bits = kBitTable[byte_val];
                    const size_t base = byte_pos * 8;
                    bit_counts[base + 0] += bits[0];
                    bit_counts[base + 1] += bits[1];
                    bit_counts[base + 2] += bits[2];
                    bit_counts[base + 3] += bits[3];
                    bit_counts[base + 4] += bits[4];
                    bit_counts[base + 5] += bits[5];
                    bit_counts[base + 6] += bits[6];
                    bit_counts[base + 7] += bits[7];
                }
            }

            // Pack the majority-vote result back into bytes.
            std::vector<uint8_t> new_bytes(d_dim, 0u);
            for (size_t byte_pos = 0; byte_pos < d_dim; ++byte_pos) {
                uint8_t new_byte = 0;
                if (bit_counts[byte_pos * 8 + 0] * 2 > count) new_byte |= 1u << 0;
                if (bit_counts[byte_pos * 8 + 1] * 2 > count) new_byte |= 1u << 1;
                if (bit_counts[byte_pos * 8 + 2] * 2 > count) new_byte |= 1u << 2;
                if (bit_counts[byte_pos * 8 + 3] * 2 > count) new_byte |= 1u << 3;
                if (bit_counts[byte_pos * 8 + 4] * 2 > count) new_byte |= 1u << 4;
                if (bit_counts[byte_pos * 8 + 5] * 2 > count) new_byte |= 1u << 5;
                if (bit_counts[byte_pos * 8 + 6] * 2 > count) new_byte |= 1u << 6;
                if (bit_counts[byte_pos * 8 + 7] * 2 > count) new_byte |= 1u << 7;
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
