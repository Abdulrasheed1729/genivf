#pragma once

#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

// this function says hello :)
void sayHello();

namespace genivf {

// A container representing a single datapoint
struct Point {
  std::vector<float> values;
  size_t id;

  Point(std::vector<float> values, size_t id)
      : values(std::move(values)), id(id) {}

  Point(std::initializer_list<float> args) : values(args) {}

  Point(const Point &) = default;
  Point(Point &&) = default;
  Point &operator=(const Point &) = default;
  Point &operator=(Point &&) = default;

  float operator[](size_t index) const;
};

struct SearchResult {
  // The point id.
  size_t id;

  // distance to the query point.
  // NOTE: need to change this to hamming (after tomorrow meeting?)
  double distance;

  // natural ordering by distance
  bool operator<(const SearchResult &other) const noexcept {
    return distance < other.distance;
  }
};

// A container representing a voronoi cell
// point_indices: the collection of the idices of point in side the cell
// centroid: the point at the centre of the cell
// id: the idex of the cell
struct Cluster {
  std::vector<size_t> point_indices;
  Point centroid;
  size_t id;

  Cluster(size_t id, Point centroid) : centroid(std::move(centroid)), id(id) {}
};

// IndexIVF — Inverted File Index for approximate nearest-neighbour search.
// num_cells: number of Voronoi cells (k in k-means).
// dim:       dimensionality of every vector in this index.
// seed:      RNG seed for centroid initialisation — fix for reproducibility,
//            vary for ensemble approaches.
struct IndexIVF {
  explicit IndexIVF(size_t num_cells, size_t dim, unsigned seed);

  explicit IndexIVF(size_t num_cells, size_t dim);

  // Run k-means on `points` to determine the cell centroids.
  // Must be called before add() or search().
  //
  // max_iter: iteration cap — increase for tighter convergence on large data.
  // epsilon:  stop early if every centroid moves less than this distance.
  void train(std::span<const Point> points, size_t max_iter = 100,
             double epsilon = 1e-6);

  // Insert points into the index. Each point is assigned to its nearest
  // centroid and stored in that cell's inverted list.
  // Throws std::invalid_argument if a duplicate id is detected, or if any
  // point's dimension does not match the index dimension.
  void add(std::span<const Point> points);

  // Return the k approximate nearest neighbours to query.
  // nprobe controls the recall/latency trade-off: searching more cells
  // improves recall but increases cost. Must satisfy 1 <= nprobe <= num_cells.
  [[nodiscard]] std::vector<SearchResult> search(const Point &query, size_t k,
                                                 size_t nprobe = 1) const;

  // True once train() has completed successfully.
  [[nodiscard]] bool is_trained() const noexcept { return !d_clusters.empty(); }

private:
  // number of voronoi cells
  size_t d_num_cells;

  // dimension of individual vectors in the inverted index
  size_t d_dim;

  // random number generator seed for centroids initialisation
  unsigned d_seed = 42;

  std::vector<Cluster> d_clusters;

  // a key-value store for each point and their id
  std::unordered_map<size_t, Point> d_vectors;

  // calculates the l2 norm, i.e. the euclidean distance between
  // `a` and `b`
  static double distance_l2(const Point &a, const Point &b);

  // calculates squared l2 norm distance between `a` and `b`
  static double distance_l2_sq(const Point &a, const Point &b);

  // returns the index into d_clusters of the centroid nearest to point.
  size_t find_nearest_centroid(const Point &point) const;
};

} // namespace genivf
