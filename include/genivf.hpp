#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

#include "utils.hpp"

namespace genivf {

struct IndexIVF;

namespace io {
void
save_index(const IndexIVF&, const std::filesystem::path&);
IndexIVF
load_index(const std::filesystem::path&);
} // namespace io

// Metric types for distance calculations.
enum class MetricType
{
    L2,
    HAMMING,
    JACCARD
};

// A container representing a single packed-binary datapoint.
//
// `id`: Unique identifier for the point within the index.
// `values`: The packed binary vector data.
struct Point
{
    size_t id;
    std::vector<uint8_t> values;

    Point(size_t id, std::vector<uint8_t> vals)
      : id(id)
      , values(std::move(vals))
    {
    }

    Point(size_t id, std::initializer_list<uint8_t> args)
      : id(id)
      , values(args)
    {
    }

    Point(const Point&) = default;
    Point(Point&&) = default;
    Point& operator=(const Point&) = default;
    Point& operator=(Point&&) = default;

    // Returns the packed byte at position `index` (not a bit index).
    // Throws `std::out_of_range` if `index` is out of range.
    [[nodiscard]] uint8_t operator[](size_t index) const;
};

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

// Simple struct representing a search result.
//
// `id`: The point id.
// `distance`: The distance to the query point.
struct SearchResult
{
    size_t id;

    double distance;

    // Natural ordering by distance (ascending).
    bool operator<(const SearchResult& other) const noexcept
    {
        return distance < other.distance;
    }
};

// A single Voronoi cell in the IVF index.
// `centroid`      — the representative binary vector for this cell.
// `point_indices` — ids of all points assigned to this cell's inverted list.
// `id`            — cell index in [0, num_cells).
struct Cluster
{
    Point centroid;
    std::vector<size_t> point_indices;
    size_t id;

    Cluster(size_t cell_id, Point ctr)
      : centroid(std::move(ctr))
      , id(cell_id)
    {
    }
};

// Binary Inverted File Index for approximate nearest-neighbour search.
//
// num_cells: number of Voronoi cells (k in k-means).
// dim:       number of bytes per vector.
// seed:      RNG seed for centroid initialisation (fix for reproducibility).
struct IndexIVF
{
    explicit IndexIVF(size_t num_cells, size_t dim, unsigned seed);

    // Constructs an IndexIVF with a default RNG seed of 42.
    explicit IndexIVF(size_t num_cells, size_t dim);

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
    // `metric`: distance metric for both centroid ranking and candidate scoring.
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

    std::vector<Cluster> d_clusters;
    std::unordered_map<size_t, Point> d_vectors;

    // Returns the index into d_clusters of the centroid with the minimum
    // Hamming distance to `point`.
    [[nodiscard]] size_t find_nearest_centroid(const Point& point) const;
};

} // namespace genivf
