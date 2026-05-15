#include "genivf.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>

void
sayHello()
{
    std::println("Hello from genivf");
}

namespace genivf {

float
Point::operator[](size_t index) const
{
    if (index >= this->values.size()) {
        throw std::out_of_range{ "the value of index is out of bound" };
    }
    return values[index];
}

IndexIVF::IndexIVF(size_t num_cells, size_t dim, unsigned seed)
  : d_num_cells(num_cells)
  , d_dim(dim)
  , d_seed(seed)
{
    if (num_cells <= 0) {
        throw std::invalid_argument("IndexIVF: num_cells must be > 0");
    }
    if (dim == 0) {
        throw std::invalid_argument("IndexIVF: dim must be > 0");
    }
}

IndexIVF::IndexIVF(size_t num_cells, size_t dim)
  : d_num_cells(num_cells)
  , d_dim(dim)
{
    if (num_cells <= 0) {
        throw std::invalid_argument("IndexIVF: num_cells must be > 0");
    }
    if (dim == 0) {
        throw std::invalid_argument("IndexIVF: dim must be > 0");
    }
}

double
IndexIVF::distance_l2_sq(const Point& a, const Point& b)
{
    assert(a.values.size() == b.values.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.values.size(); ++i) {
        const double diff =
          static_cast<double>(a.values[i]) - static_cast<double>(b.values[i]);
        sum += diff * diff;
    }
    return sum;
}

double
IndexIVF::distance_l2(const Point& a, const Point& b)
{
    return std::sqrt(distance_l2_sq(a, b));
}

size_t
IndexIVF::find_nearest_centroid(const Point& point) const
{
    size_t nearest = 0;
    double min_dist_sq = distance_l2_sq(point, this->d_clusters[0].centroid);

    for (size_t i = 1; i < this->d_clusters.size(); ++i) {
        const double dist_sq =
          distance_l2_sq(point, this->d_clusters[i].centroid);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            nearest = i;
        }
    }
    return nearest;
}

void
IndexIVF::train(std::span<const Point> points, size_t max_iter, double epsilon)
{
    if (points.size() < this->d_num_cells) {
        throw std::invalid_argument(
          "IndexIVF::train: need at least num_cells training points");
    }

    // confirms the dimension of every data points
    // NOTE: is this really necessary?
    for (const auto& p : points) {
        if (p.values.size() != d_dim) {
            throw std::invalid_argument("IndexIVF::train: point dimension does "
                                        "not match index dimension");
        }
    }

    // HACK: Clear both stores so re-training is safe.
    this->d_clusters.clear();
    this->d_vectors.clear();

    const size_t n = points.size();
    // HACK: Squaring epsilon avoids a sqrt in the convergence check.
    const double epsilon_sq = epsilon * epsilon;

    // NOTE: initialise centroids by randomly sampling k distinct points
    std::mt19937 rng(d_seed);
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<Point> centroids;
    centroids.reserve(this->d_num_cells);
    for (size_t i = 0; i < this->d_num_cells; ++i) {
        centroids.emplace_back(points[idx[i]].values, i);
    }

    // NOTE: Each entry holds the indices (into `points`) assigned to cluster i.
    std::vector<std::vector<size_t>> assignments(this->d_num_cells);

    // HACK: Pre-allocate new_centroids outside the loop — avoids reallocating
    // the outer vector on every iteration.
    std::vector<Point> new_centroids;
    new_centroids.reserve(this->d_num_cells);

    // iterate until convergence
    for (size_t iter = 0; iter < max_iter; ++iter) {

        // clear out every assignments
        for (auto& a : assignments) {
            a.clear();
        }

        // assign each point to its nearest centroid.
        // no sqrt is needed for arg min ||x_p - c_i||^2
        for (size_t i = 0; i < n; ++i) {
            size_t nearest = 0;
            double min_dist_sq = distance_l2_sq(points[i], centroids[0]);

            for (size_t j = 1; j < d_num_cells; ++j) {
                const double dist_sq = distance_l2_sq(points[i], centroids[j]);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    nearest = j;
                }
            }
            assignments[nearest].push_back(i);
        }

        // Recompute each centroid as the mean of its assigned points.
        bool converged = true;
        new_centroids.clear(); // preserves capacity

        for (size_t i = 0; i < this->d_num_cells; ++i) {
            if (assignments[i].empty()) {
                // A cluster lost all its members — keep the old centroid in
                // place. Alternative: re-seed from the highest-error point.
                new_centroids.push_back(centroids[i]);
                continue;
            }

            std::vector<float> mean(d_dim, 0.0f);
            for (const size_t pt_idx : assignments[i]) {
                for (size_t d = 0; d < d_dim; ++d) {
                    mean[d] += points[pt_idx].values[d];
                }
            }

            const float count = static_cast<float>(assignments[i].size());
            for (auto& val : mean) {
                val /= count;
            }

            Point new_centroid(std::move(mean), i);

            // Use squared distance so we avoid sqrt in the convergence check.
            // Compare against epsilon^2 — same threshold, no transcendental
            // call.
            if (distance_l2_sq(centroids[i], new_centroid) > epsilon_sq) {
                converged = false;
            }

            new_centroids.push_back(std::move(new_centroid));
        }

        centroids = std::move(new_centroids);

        if (converged) {
            break;
        }
    }

    // materialise clusters with empty inverted lists
    this->d_clusters.reserve(d_num_cells);
    for (size_t i = 0; i < this->d_num_cells; ++i) {
        this->d_clusters.emplace_back(i, std::move(centroids[i]));
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

        // Detect duplicate IDs before inserting the emplace() silently no-ops
        // on a collision, which would leave the inverted list pointing at the
        // old vector.
        auto [it, inserted] = d_vectors.emplace(point.id, point);
        if (!inserted) {
            throw std::invalid_argument("IndexIVF::add: duplicate point id " +
                                        std::to_string(point.id));
        }

        const size_t cell = this->find_nearest_centroid(point);
        this->d_clusters[cell].point_indices.push_back(point.id);
    }
}

std::vector<SearchResult>
IndexIVF::search(const Point& query, size_t k, size_t nprobe) const
{
    if (!this->is_trained()) {
        throw std::logic_error(
          "IndexIVF::search: call train() before search()");
    }
    if (nprobe == 0 || nprobe > this->d_clusters.size()) {
        throw std::invalid_argument(
          "IndexIVF::search: nprobe must be in [1, num_cells]");
    }
    if (query.values.size() != d_dim) {
        throw std::invalid_argument(
          "IndexIVF::search: query dimension does not match index dimension");
    }

    // rank all centroids by squared distance to the query
    // Squared distance is sufficient for ordering; we don't need the real
    // distance to select which cells to probe.
    std::vector<std::pair<double, size_t>> centroid_dists;
    centroid_dists.reserve(this->d_clusters.size());

    for (size_t i = 0; i < d_clusters.size(); ++i) {
        centroid_dists.emplace_back(
          distance_l2_sq(query, this->d_clusters[i].centroid), i);
    }

    // partial_sort is O(n log nprobe) — cheaper than a full sort when nprobe
    // is small relative to the total number of cells.
    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() +
                        static_cast<std::ptrdiff_t>(nprobe),
                      centroid_dists.end());

    // scan the nprobe nearest inverted lists
    // Use the real Euclidean distance here. this is what gets returned to the
    // caller as SearchResult.distance.
    std::vector<SearchResult> candidates;

    for (size_t p = 0; p < nprobe; ++p) {
        const size_t cell_idx = centroid_dists[p].second;

        for (const size_t point_id : this->d_clusters[cell_idx].point_indices) {
            const Point& pt = d_vectors.at(point_id);
            candidates.push_back({ point_id, distance_l2(query, pt) });
        }
    }

    // sort the top-k results by distance
    const size_t result_count = std::min(k, candidates.size());

    std::partial_sort(candidates.begin(),
                      candidates.begin() +
                        static_cast<std::ptrdiff_t>(result_count),
                      candidates.end());

    candidates.resize(result_count);
    return candidates;
}

} // namespace genivf
