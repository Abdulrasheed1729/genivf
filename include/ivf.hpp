#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

#include "types.hpp"
#include "utils.hpp"

namespace genivf {

struct IndexIVF;

namespace io {
void
save_index(const IndexIVF&, const std::filesystem::path&);
IndexIVF
load_index(const std::filesystem::path&);
} // namespace io

[[nodiscard]] inline uint32_t
distance_hamming(const Point& a, const Point& b)
{
    assert(a.values.size() == b.values.size());
    return distance_hamming(a.values.data(), b.values.data(), a.values.size());
}

[[nodiscard]] inline float
distance_jaccard(const Point& a, const Point& b)
{
    assert(a.values.size() == b.values.size());
    return distance_jaccard(a.values.data(), b.values.data(), a.values.size());
}

[[nodiscard]] inline double
distance_l2(const Point& a, const Point& b)
{
    assert(a.values.size() == b.values.size());
    return distance_l2(a.values.data(), b.values.data(), a.values.size());
}

[[nodiscard]] inline double
distance_l2_sq(const Point& a, const Point& b)
{
    assert(a.values.size() == b.values.size());
    return distance_l2_sq(a.values.data(), b.values.data(), a.values.size());
}

// Binary Inverted File Index for approximate nearest-neighbour search.
//
// num_cells: number of Voronoi cells (k in k-means).
// dim:       number of bytes per vector.
// seed:      RNG seed for centroid initialisation (fix for reproducibility).
struct IndexIVF
{
    explicit IndexIVF(size_t num_cells,
                      size_t dim,
                      unsigned seed,
                      InitType init = InitType::KMEANS_PLUS_PLUS);

    // Constructs an IndexIVF with a default RNG seed of 42.
    explicit IndexIVF(size_t num_cells,
                      size_t dim,
                      InitType init = InitType::KMEANS_PLUS_PLUS);

    void train(std::span<const Point> points,
               size_t max_iter = 100,
               double epsilon = 1e-6);

    // Insert points into the index. Each point is assigned to the centroid
    // with the smallest Hamming distance and stored in that cell's inverted
    // list.
    //
    // Throws `std::logic_error` if `train()` has not been called.
    // Throws `std::invalid_argument` on dimension mismatch or duplicate id.
    void add(std::span<const Point> points);

    // Return the k approximate nearest neighbours to `query`.
    //
    // `nprobe`: number of Voronoi cells to probe. Higher values increase recall
    //           at the cost of latency. Must satisfy 1 <= nprobe <= num_cells.
    // `metric`: distance metric for both centroid ranking and candidate
    // scoring.
    //           For best recall use HAMMING, which matches the training metric.
    //
    // Throws `std::logic_error` if `train()` has not been called.
    // Throws `std::invalid_argument` on bad `nprobe` or dimension mismatch.
    [[nodiscard]] std::vector<SearchResult> search(
      const Point& query,
      size_t k,
      size_t nprobe = 1,
      MetricType metric = MetricType::HAMMING) const;

    // True once train() has completed successfully.
    [[nodiscard]] bool is_trained() const noexcept
    {
        return !d_clusters.empty();
    }

    // IO functions need direct access to private members to serialise and
    // reconstruct the index without exposing them through the public API.
    friend void io::save_index(const IndexIVF&, const std::filesystem::path&);
    friend IndexIVF io::load_index(const std::filesystem::path&);

  private:
    size_t d_num_cells;
    size_t d_dim; // number of bytes per vector (bit-width / 8)
    unsigned d_seed = 42;
    InitType d_init_type = InitType::RANDOM;

    std::vector<Cluster> d_clusters;
    std::unordered_map<size_t, Point> d_vectors;

    // Returns the index into d_clusters of the centroid with the minimum
    // Hamming distance to `point`.
    [[nodiscard]] size_t find_nearest_centroid(const Point& point) const;

    template<MetricType Metric>
    [[nodiscard]] std::vector<SearchResult> search_impl(const Point& query,
                                                        size_t k,
                                                        size_t nprobe) const;
};

} // namespace genivf
